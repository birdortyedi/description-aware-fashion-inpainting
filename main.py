import torch
from torch import nn, optim
from torch.utils import data
from torchvision.transforms import ToTensor, ToPILImage, Resize
from torchvision.transforms import functional as F
from tensorboardX import SummaryWriter

from tqdm import tqdm
from colorama import Fore

from utils import HDF5Dataset, UnNormalize, weights_init, unnormalize_img
from models import CoarseNet, RefineNet, LocalDiscriminator, GlobalDiscriminator, VGG16
from losses import CoarseLoss, RefineLoss

NUM_EPOCHS = 250
BATCH_SIZE = 128

fg_train = HDF5Dataset(filename='./Fashion-Gen/fashiongen_256_256_train.h5')
fg_val = HDF5Dataset(filename='./Fashion-Gen/fashiongen_256_256_validation.h5', is_train=False)

print("Sample size in training: {}".format(len(fg_train)))

train_loader = data.DataLoader(fg_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = data.DataLoader(fg_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
coarse = CoarseNet(fg_train.vocab_size)
refine = RefineNet()
local_d = LocalDiscriminator()
global_d = GlobalDiscriminator()
# vgg = VGG16(requires_grad=False)
if torch.cuda.device_count() > 1:
    print("Using {} GPUs...".format(torch.cuda.device_count()))
    coarse = nn.DataParallel(coarse)
    refine = nn.DataParallel(refine)
    local_d = nn.DataParallel(local_d)
    global_d = nn.DataParallel(global_d)
coarse.to(device)
refine.to(device)
local_d.to(device)
global_d.to(device)
# vgg.to(device)

coarse.apply(weights_init)
refine.apply(weights_init)
local_d.apply(weights_init)
global_d.apply(weights_init)

coarse_loss_fn = CoarseLoss()
coarse_loss_fn = coarse_loss_fn.to(device)
refine_loss_fn = RefineLoss()
refine_loss_fn = refine_loss_fn.to(device)

d_loss_fn = nn.BCELoss()
d_loss_fn = d_loss_fn.to(device)

c_lr, r_lr, d_lr = 0.002, 0.001, 0.0001
coarse_optimizer = optim.Adam(coarse.parameters(), lr=c_lr, betas=(0.5, 0.999))
refine_optimizer = optim.Adam(refine.parameters(), lr=r_lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(local_d.parameters(), lr=d_lr, betas=(0.9, 0.999))

coarse_scheduler = optim.lr_scheduler.ExponentialLR(coarse_optimizer, gamma=0.9)
refine_scheduler = optim.lr_scheduler.ExponentialLR(refine_optimizer, gamma=0.95)
d_scheduler = optim.lr_scheduler.ExponentialLR(d_optimizer, gamma=0.95)

writer = SummaryWriter()


def train(epoch, loader, l_fns, optimizers, schedulers):
    coarse.train()
    refine.train()
    for batch_idx, (x_train, x_desc, x_mask, x_local, local_coords, y_train) in tqdm(enumerate(loader), ncols=50, desc="Training",
                                                                                     bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
        num_step = epoch * len(loader) + batch_idx
        x_train = x_train.float().to(device)
        x_desc = x_desc.long().to(device)
        x_mask = x_mask.float().to(device)
        x_local = x_local.float().to(device)
        y_train = y_train.float().to(device)

        coarse_output = train_coarse(num_step, x_train, x_desc, y_train, l_fns)
        optimizers["coarse"].step()
        schedulers["coarse"].step(epoch)

        refine_output, refine_local_output, refine_losses = train_refine(num_step, coarse_output, x_mask, y_train, local_coords, l_fns)
        optimizers["refine"].step()
        schedulers["refine"].step(epoch)

        train_discriminator(num_step, x_train[BATCH_SIZE//8, :, :, :], x_desc[BATCH_SIZE//8, :],
                            x_mask[BATCH_SIZE//8, :, :, :], x_local[BATCH_SIZE//8, :, :, :], y_train[BATCH_SIZE//8, :, :, :],
                            local_coords[BATCH_SIZE//8, :], l_fns)
        optimizers["discriminator"].step()
        schedulers["discriminator"].step(epoch)

        global_d_accuracy_on_refine_output = torch.mean((global_d(refine_output).view(-1) > 0.5).float(), dim=0)
        writer.add_scalar("Metrics/on_step_d_global_acc_on_refine", global_d_accuracy_on_refine_output, num_step)
        local_d_accuracy_on_refine_local_output = torch.mean((local_d(refine_local_output).view(-1) > 0.5).float(), dim=0)
        writer.add_scalar("Metrics/on_step_d_local_acc_on_refine", local_d_accuracy_on_refine_local_output, num_step)

        if batch_idx % 50 == 0:
            make_verbose(x_train, x_local, y_train, coarse_output, refine_output, refine_local_output, refine_losses, num_step, batch_idx, epoch)


def train_refine(num_step, coarse_output, x_mask, y_train, local_coords, l_fns):
    refine.zero_grad()
    refine_output = (1.0 - x_mask) * y_train + x_mask * refine(coarse_output)
    # refine_output_vgg_features = vgg(normalize_batch(refine_output))
    refine_local_output = list()
    for im, local_coord in zip(refine_output, local_coords):
        top, left, h, w = local_coord
        single_out = ToTensor()(Resize(size=(32, 32))(F.crop(ToPILImage()(im.cpu()), top.item(), left.item(), h.item(), w.item())))
        refine_local_output.append(single_out)
    refine_local_output = torch.stack(refine_local_output).to(device)

    real_label = torch.ones((y_train.size(0), 1)).to(device)
    refine_global_loss = loss_fns["discriminator"](global_d(refine_output), real_label)
    refine_local_loss = loss_fns["discriminator"](local_d(refine_local_output), real_label)
    refine_loss, refine_pixel, refine_style, refine_tv = l_fns["refine"](refine_output, y_train)  # , coarse_output_vgg_features, refine_output_vgg_features)
    writer.add_scalar("Loss/on_step_refine_loss", refine_loss.mean().item(), num_step)
    writer.add_scalar("Loss/on_step_refine_pixel_loss", refine_pixel.mean().item(), num_step)
    # writer.add_scalar("Loss/on_step_refine_content_loss", refine_content.mean().item(), num_step)
    writer.add_scalar("Loss/on_step_refine_style_loss", refine_style.mean().item(), num_step)
    writer.add_scalar("Loss/on_step_refine_tv_loss", refine_tv.mean().item(), num_step)
    writer.add_scalar("Loss/on_step_refine_global_loss", refine_global_loss.mean().item(), num_step)
    writer.add_scalar("Loss/on_step_refine_local_loss", refine_local_loss.mean().item(), num_step)
    loss = (0.4 * refine_global_loss + 0.6 * refine_local_loss) + (2.0 * refine_loss)
    loss.backward()

    return refine_output, refine_local_output, (refine_loss, refine_pixel, refine_style, refine_tv, refine_global_loss, refine_local_loss)


def train_discriminator(num_step, x_train, x_desc, x_mask, x_local, y_train, local_coords, l_fns):
    global_d.zero_grad()
    global_d_real_output = global_d(y_train).view(-1)
    real_label = torch.ones_like(global_d_real_output).to(device)
    global_real_loss = l_fns["discriminator"](global_d_real_output, real_label)
    writer.add_scalar("Loss/on_step_d_global_real_loss", global_real_loss.mean().item(), num_step)
    global_fake_output = (1.0 - x_mask) * x_train + x_mask * refine(coarse(x_train, x_desc))
    global_d_fake_output = global_d(global_fake_output).view(-1)
    fake_label = torch.zeros_like(global_d_fake_output).to(device)
    global_fake_loss = l_fns["discriminator"](global_d_fake_output, fake_label)
    writer.add_scalar("Loss/on_step_d_global_fake_loss", global_fake_loss.mean().item(), num_step)
    global_loss = global_real_loss + global_fake_loss

    local_d.zero_grad()
    local_d_real_output = local_d(x_local).view(-1)
    local_real_loss = l_fns["discriminator"](local_d_real_output, real_label)
    writer.add_scalar("Loss/on_step_d_local_real_loss", local_real_loss.mean().item(), num_step)
    out_local = list()
    for im, local_coord in zip(global_fake_output, local_coords):
        top, left, h, w = local_coord
        single_out = ToTensor()(Resize(size=(32, 32))(F.crop(ToPILImage()(im.cpu()), top.item(), left.item(), h.item(), w.item())))
        out_local.append(single_out)
    out_local = torch.stack(out_local).to(device)
    local_d_fake_output = local_d(out_local).view(-1)
    local_fake_loss = l_fns["discriminator"](local_d_fake_output, fake_label)
    writer.add_scalar("Loss/on_step_d_local_fake_loss", local_fake_loss.mean().item(), num_step)
    local_loss = local_real_loss + local_fake_loss
    d_loss = global_loss + local_loss
    d_loss.backward(retain_graph=True)


def train_coarse(num_step, x_train, x_desc, y_train, l_fns):
    coarse.zero_grad()
    coarse_output = coarse(x_train, x_desc)
    # coarse_vgg_features = vgg(normalize_batch(x_train))
    # coarse_output_vgg_features = vgg(normalize_batch(coarse_output))
    coarse_loss, coarse_pixel, coarse_style = l_fns["coarse"](coarse_output, y_train)  # , coarse_vgg_features, coarse_output_vgg_features)

    writer.add_scalar("Loss/on_step_coarse_loss", coarse_loss.mean().item(), num_step)
    writer.add_scalar("Loss/on_step_coarse_pixel_loss", coarse_pixel.mean().item(), num_step)
    # writer.add_scalar("Loss/on_step_coarse_content_loss", coarse_content.mean().item(), num_step)
    writer.add_scalar("Loss/on_step_coarse_style_loss", coarse_style.mean().item(), num_step)

    coarse_loss.backward(retain_graph=True)

    return coarse_output  # , coarse_output_vgg_features


def make_verbose(x_train, x_local, y_train, coarse_output, refine_output, refine_local_output, refine_losses, num_step, batch_idx, epoch):
    # unnormalizer = UnNormalize((0.7535, 0.7359, 0.7292), (0.5259, 0.5487, 0.5589))
    x_0 = (unnormalize_img(x_train[0]).cpu()).detach().numpy()  # UnNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    y_0 = (unnormalize_img(y_train[0]).cpu()).detach().numpy()
    local_0 = (unnormalize_img(x_local[0]).cpu()).detach().numpy()
    coarse_0 = (unnormalize_img(coarse_output[0]).squeeze(0).cpu()).detach().numpy()
    refine_0 = (unnormalize_img(refine_output[0]).squeeze(0).cpu()).detach().numpy()
    refine_local_0 = (unnormalize_img(refine_local_output[0]).squeeze(0).cpu()).detach().numpy()
    writer.add_image("train_x/epoch_{}".format(epoch), x_0, num_step)
    writer.add_image("original/epoch_{}".format(epoch), y_0, num_step)
    writer.add_image("local_x/epoch_{}".format(epoch), local_0, num_step)
    writer.add_image("coarse_out/epoch_{}".format(epoch), coarse_0, num_step)
    writer.add_image("refine_out/epoch_{}".format(epoch), refine_0, num_step)
    writer.add_image("refine_local_out/epoch_{}".format(epoch), refine_local_0, num_step)

    refine_loss, refine_pixel, refine_style, refine_tv, refine_global, refine_local = refine_losses
    print("Step:{}  ".format(num_step),
          "Epoch:{}".format(epoch),
          "[{}/{} ".format(batch_idx * len(x_train), len(train_loader.dataset)),
          "({}%)]  ".format(int(100 * batch_idx / float(len(train_loader)))),
          "Loss: {:.6f} ".format(refine_loss.mean().item()),
          "Pixel: {:.6f} ".format(refine_pixel.mean().item()),
          # "Content: {:.5f} ".format(refine_content.mean().item()),
          "Style: {:.6f} ".format(refine_style.mean().item()),
          "TV: {:.6f} ".format(refine_tv.mean().item()),
          "Global: {:.6f} ".format(refine_global.mean().item()),
          "Local: {:.6f}".format(refine_local.mean().item()))


def evaluate(epoch, loader, l_fn):
    num_step = 0
    total_loss, total_content_loss, total_style_loss, total_struct_loss, total_adverserial_loss = 0., 0., 0., 0., 0.
    with torch.no_grad():
        net.eval()
        for batch_idx, (x_val, x_desc_val, y_val) in tqdm(enumerate(loader), ncols=50, desc="Validation",
                                                          bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
            num_step += 1
            x_val = x_val.float().to(device)
            x_desc_val = x_desc_val.long().to(device)
            y_val = y_val.float().to(device)

            output = net(x_val, x_desc_val)
            d_output = d_net(output).view(-1)
            val_loss, val_content, val_style, val_struct, val_adversarial = l_fn[1](output, y_val, d_output)

            total_loss += val_loss.item()
            total_content_loss += val_content.item()
            total_style_loss += val_style.item()
            total_struct_loss += val_struct.item()
            total_adverserial_loss += val_adversarial.item()

            if batch_idx % 50 == 0:
                print("[{}/{} ".format(batch_idx * len(x_val), len(loader.dataset)),
                      "({}%)]\t".format(int(100 * batch_idx / float(len(loader)))),
                      "Loss: {:.4f}".format(val_loss.item()),
                      "Content: {:.4f}  ".format(val_content.item()),
                      "Style: {:.9f}  ".format(val_style.item()),
                      "Structure: {:.4f}  ".format(val_struct.item()),
                      "Adversarial: {:.4f}  ".format(val_adversarial.item()))

        writer.add_scalar("Loss/on_epoch_val_loss", total_loss / num_step, epoch)
        writer.add_scalar("Loss/on_epoch_val_content_loss", total_content_loss / num_step, epoch)
        writer.add_scalar("Loss/on_epoch_val_style_loss", total_style_loss / num_step, epoch)
        writer.add_scalar("Loss/on_epoch_val_structure_loss", total_struct_loss / num_step, epoch)
        writer.add_scalar("Loss/on_epoch_val_adversarial_loss", total_adverserial_loss / num_step, epoch)


if __name__ == '__main__':
    loss_fns = {"coarse": coarse_loss_fn,
                "refine": refine_loss_fn,
                "discriminator": d_loss_fn}
    optimizers = {"coarse": coarse_optimizer,
                  "refine": refine_optimizer,
                  "discriminator": d_optimizer}
    schedulers = {"coarse": coarse_scheduler,
                  "refine": refine_scheduler,
                  "discriminator": d_scheduler}
    for e in range(NUM_EPOCHS):
        train(e, train_loader, loss_fns, optimizers, schedulers)
        # evaluate(e, val_loader, (d_loss_fn, loss_fn))
        torch.save(coarse.state_dict(), "./weights/{}/weights_epoch_{}.pth".format("coarse", e))
        torch.save(refine.state_dict(), "./weights/{}/weights_epoch_{}.pth".format("refine", e))
