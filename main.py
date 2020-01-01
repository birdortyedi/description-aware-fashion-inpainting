import torch
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from tensorboardX import SummaryWriter

from tqdm import tqdm
from colorama import Fore

from utils import HDF5Dataset, RandomCentralErasing, UnNormalize
from models import CoarseNet, RefineNet, LocalDiscriminator, GlobalDiscriminator
from losses import CustomInpaintingLoss

NUM_EPOCHS = 250
BATCH_SIZE = 256


train_transform = transforms.Compose([transforms.ToTensor(),
                                      RandomCentralErasing(p=1.0, scale=(0.0625, 0.125), ratio=(0.75, 1.25), value=1),
                                      # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                      ])

val_transform = transforms.Compose([transforms.ToTensor(),
                                    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])

fg_train = HDF5Dataset(filename='./Fashion-Gen/fashiongen_256_256_train.h5', transform=train_transform)
fg_val = HDF5Dataset(filename='./Fashion-Gen/fashiongen_256_256_validation.h5', transform=val_transform)

print("Sample size in training: {}".format(len(fg_train)))

train_loader = data.DataLoader(fg_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = data.DataLoader(fg_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = CoarseNet(fg_train.vocab_size)  # RefineNet()   # Net(fg_train.vocab_size)
d_net = LocalDiscriminator()
if torch.cuda.device_count() > 1:
    print("Using {} GPUs...".format(torch.cuda.device_count()))
    net = nn.DataParallel(net)
    d_net = nn.DataParallel(d_net)
net.to(device)
d_net.to(device)

loss_fn = CustomInpaintingLoss()
loss_fn = loss_fn.to(device)

d_loss_fn = nn.BCELoss()
d_loss_fn = d_loss_fn.to(device)

lr = 0.0002
optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(d_net.parameters(), lr=lr, betas=(0.5, 0.999))
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
d_scheduler = optim.lr_scheduler.ExponentialLR(d_optimizer, gamma=0.95)

writer = SummaryWriter()


def train(epoch, loader, l_fns, opts, schs):
    net.train()
    num_step = 0
    total_d_loss, total_real_loss, total_fake_loss = 0., 0., 0.
    total_loss, total_content_loss, total_style_loss, total_struct_loss, total_adverserial_loss = 0., 0., 0., 0., 0.
    for batch_idx, (x_train, x_desc, y_train) in tqdm(enumerate(loader), ncols=50, desc="Training",
                                                      bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
        num_step += 1
        d_net.zero_grad()

        x_train = x_train.float().to(device)
        x_desc = x_desc.long().to(device)
        y_train = y_train.float().to(device)

        d_real_output = d_net(x_train).view(-1)
        real_label = torch.ones_like(d_real_output).to(device)
        real_loss = l_fns[0](d_real_output, real_label)
        real_loss.backward()
        writer.add_scalar("Loss/on_step_discriminator_real_loss", real_loss.mean().item(), epoch * len(loader) + batch_idx)

        fake_inputs = net(x_train, x_desc)
        d_fake_output = d_net(fake_inputs).view(-1)
        fake_label = torch.zeros_like(d_fake_output).to(device)
        fake_loss = l_fns[0](d_fake_output, fake_label)
        fake_loss.backward()
        writer.add_scalar("Loss/on_step_discriminator_fake_loss", fake_loss.mean().item(), epoch * len(loader) + batch_idx)

        d_loss = real_loss + fake_loss
        total_d_loss += d_loss.item()
        total_real_loss += real_loss.item()
        total_fake_loss += fake_loss.item()
        writer.add_scalar("Loss/on_step_discriminator_total_loss", d_loss.mean().item(), epoch * len(loader) + batch_idx)

        opts[0].step()
        schs[0].step(epoch)

        net.zero_grad()
        output = net(x_train, x_desc)
        d_output = d_net(output).view(-1)
        d_accuracy = torch.mean((d_output > 0.5).float(), dim=0)
        writer.add_scalar("Metrics/on_step_d_acc_on_g", d_accuracy, epoch * len(loader) + batch_idx)
        loss, content, style, struct, adversarial = l_fns[1](output, y_train, d_output)
        loss.backward()

        total_loss += loss.item()
        total_content_loss += content.item()
        total_style_loss += style.item()
        total_struct_loss += struct.item()
        total_adverserial_loss += adversarial.item()

        opts[1].step()
        schs[1].step(epoch)

        writer.add_scalar("Loss/on_step_loss", loss.item(), epoch * len(loader) + batch_idx)
        writer.add_scalar("Loss/on_step_content_loss", content.item(), epoch * len(loader) + batch_idx)
        writer.add_scalar("Loss/on_step_style_loss", style.item(), epoch * len(loader) + batch_idx)
        writer.add_scalar("Loss/on_step_structure_loss", struct.item(), epoch * len(loader) + batch_idx)
        writer.add_scalar("Loss/on_step_adversarial_loss", adversarial.item(), epoch * len(loader) + batch_idx)

        if batch_idx % 50 == 0:
            num_step = epoch * len(loader) + batch_idx
            x_0 = (x_train[0].cpu()).detach().numpy()  # UnNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            y_0 = (y_train[0].cpu()).detach().numpy()
            out_0 = (output[0].squeeze(0).cpu()).detach().numpy()
            writer.add_image("train_x/epoch_{}".format(epoch), x_0, num_step)
            writer.add_image("original_x/epoch_{}".format(epoch), y_0, num_step)
            writer.add_image("output/epoch_{}".format(epoch), out_0, num_step)
            print("Step: {}\t".format(num_step),
                  "Epoch: {}".format(epoch),
                  "[{}/{} ".format(batch_idx * len(x_train), len(train_loader.dataset)),
                  "({}%)]\t".format(int(100 * batch_idx / float(len(train_loader)))),
                  "Loss: {:.4f}  ".format(loss.item()),
                  "Content: {:.4f}  ".format(content.item()),
                  "Style: {:.4f}  ".format(style.item()),
                  "Structure: {:.4f}  ".format(struct.item()),
                  "Adversarial: {:.4f}  ".format(adversarial.item()))

    writer.add_scalar("Loss/on_epoch_d_total_loss", total_d_loss / num_step, epoch)
    writer.add_scalar("Loss/on_epoch_d_real_loss", total_real_loss / num_step, epoch)
    writer.add_scalar("Loss/on_epoch_d_fake_loss", total_fake_loss / num_step, epoch)
    writer.add_scalar("Loss/on_epoch_loss", total_loss / num_step, epoch)
    writer.add_scalar("Loss/on_epoch_content_loss", total_content_loss / num_step, epoch)
    writer.add_scalar("Loss/on_epoch_style_loss", total_style_loss / num_step, epoch)
    writer.add_scalar("Loss/on_epoch_structure_loss", total_struct_loss / num_step, epoch)
    writer.add_scalar("Loss/on_epoch_adverserial_loss", total_adverserial_loss / num_step, epoch)


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
    for e in range(NUM_EPOCHS):
        train(e, train_loader, (d_loss_fn, loss_fn), (d_optimizer, optimizer), (d_scheduler, scheduler))
        evaluate(e, val_loader, (d_loss_fn, loss_fn))
        torch.save(net.state_dict(), "./weights/weights_epoch_{}.pth".format(e))
