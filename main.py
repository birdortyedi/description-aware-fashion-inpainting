import torch
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from tqdm import tqdm
from colorama import Fore
from math import log10

import os

from pytorch_msssim import SSIM
from utils import HDF5Dataset, weights_init, normalize_batch, unnormalize_batch
from models import Net, BaseNet, Discriminator, VGG16
from losses import CustomLoss, RefineLoss

NUM_EPOCHS = 5
BATCH_SIZE = 32
TEST = True

fg_train = HDF5Dataset(filename='./Fashion-Gen/fashiongen_256_256_train.h5')
fg_val = HDF5Dataset(filename='./Fashion-Gen/fashiongen_256_256_validation.h5', is_train=False)

print("Sample size in training: {}".format(len(fg_train)))

train_img_loader = data.DataLoader(fg_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_img_loader = data.DataLoader(fg_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

mask_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.RandomVerticalFlip(p=0.5),
                                     transforms.Resize(256),
                                     transforms.ToTensor(), ])

m_train = ImageFolder(root="./qd_imd/train/", transform=mask_transform)
m_val = ImageFolder(root="./qd_imd/test/", transform=mask_transform)

print("Mask size in training: {}".format(len(m_train)))

train_mask_loader = data.DataLoader(m_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_mask_loader = data.DataLoader(m_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d_net = Discriminator()
net = BaseNet(vocab_size=fg_train.vocab_size, attention=False)
refine_net = BaseNet(attention=False, lstm=False)
vgg = VGG16(requires_grad=False)
if torch.cuda.device_count() > 1:
    print("Using {} GPUs...".format(torch.cuda.device_count()))
    d_net = nn.DataParallel(d_net).to(device)
    net = nn.DataParallel(net).to(device)
    refine_net = nn.DataParallel(refine_net).to(device)
vgg.to(device)

net.apply(weights_init)
refine_net.apply(weights_init)

d_loss_fn = nn.BCELoss()
d_loss_fn = d_loss_fn.to(device)
loss_fn = CustomLoss()
loss_fn = loss_fn.to(device)
refine_loss_fn = RefineLoss()
refine_loss_fn = refine_loss_fn.to(device)

lr, r_lr, d_lr = 0.0002, 0.0002, 0.0001
d_optimizer = optim.Adam(d_net.parameters(), lr=d_lr, betas=(0.9, 0.999))
r_optimizer = optim.Adam(refine_net.parameters(), lr=r_lr, betas=(0.5, 0.999))
optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.0, 0.999))

d_scheduler = optim.lr_scheduler.ExponentialLR(d_optimizer, gamma=0.95)
r_scheduler = optim.lr_scheduler.ExponentialLR(r_optimizer, gamma=0.95)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

writer = SummaryWriter()


def train(epoch, img_loader, mask_loader):
    for batch_idx, (y_train, x_desc, x_train, x_mask) in tqdm(enumerate(img_loader), ncols=50, desc="Training",
                                             bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
        # x_mask, _ = next(iter(mask_loader))
        # if x_mask.size(0) != y_train.size(0):
        #    x_mask = x_mask[:y_train.size(0)]
        num_step = epoch * len(img_loader) + batch_idx
        x_train = x_train.float().to(device)
        x_desc = x_desc.long().to(device)
        x_mask = x_mask.float().to(device)
        # x_local = x_local.float().to(device)
        y_train = y_train.float().to(device)

        # x_train = x_mask * y_train + (1.0 - x_mask) * 0.5
        noise = torch.zeros((x_mask.size(0), 256), dtype=torch.float32).normal_().to(device)

        net.zero_grad()
        output = net(x_train, x_mask, x_desc, noise)
        composite = x_mask * y_train + (1.0 - x_mask) * output

        vgg_features_gt = vgg(normalize_batch(unnormalize_batch(y_train)))
        vgg_features_composite = vgg(composite)
        vgg_features_output = vgg(output)

        total_loss, pixel_valid_loss, pixel_hole_loss,\
            content_loss, style_loss, tv_loss = loss_fn(y_train, output, composite, x_mask,
                                                        vgg_features_gt, vgg_features_composite, vgg_features_output)

        writer.add_scalar("Coarse_G/on_step_total_loss", total_loss.item(), num_step)
        writer.add_scalar("Coarse_G/on_step_pixel_valid_loss", pixel_valid_loss.item(), num_step)
        writer.add_scalar("Coarse_G/on_step_pixel_hole_loss", pixel_hole_loss.item(), num_step)
        writer.add_scalar("Coarse_G/on_step_content_loss", content_loss.item(), num_step)
        writer.add_scalar("Coarse_G/on_step_style_loss", style_loss.item(), num_step)
        writer.add_scalar("Coarse_G/on_step_tv_loss", tv_loss.item(), num_step)
        writer.add_scalar("LR/learning_rate", scheduler.get_lr(), num_step)

        total_loss.backward()
        optimizer.step()

        refine_net.zero_grad()
        noise = torch.zeros((x_mask.size(0), 256), dtype=torch.float32).normal_().to(device)
        r_output = refine_net(output.detach(), noise=noise)
        d_output = d_net(r_output.detach()).view(-1)
        r_composite = x_mask * y_train + (1.0 - x_mask) * r_output

        r_total_loss, r_pixel_loss, adversarial_loss = refine_loss_fn(y_train, r_output, r_composite, d_output)
        writer.add_scalar("Refine_G/on_step_total_loss", r_total_loss.item(), num_step)
        writer.add_scalar("Refine_G/on_step_pixel_loss", r_pixel_loss.item(), num_step)
        writer.add_scalar("Refine_G/on_step_adversarial_loss", adversarial_loss.item(), num_step)

        r_total_loss.backward()
        r_optimizer.step()

        d_net.zero_grad()
        d_real_output = d_net(y_train).view(-1)
        d_fake_output = d_output.detach()

        if torch.rand(1) > 0.1:
            d_real_loss = d_loss_fn(d_real_output, torch.FloatTensor(d_real_output.size(0)).uniform_(0.0, 0.3).to(device))
            d_fake_loss = d_loss_fn(d_fake_output, torch.FloatTensor(d_fake_output.size(0)).uniform_(0.7, 1.2).to(device))
        else:
            d_real_loss = d_loss_fn(d_real_output, torch.FloatTensor(d_fake_output.size(0)).uniform_(0.7, 1.2).to(device))
            d_fake_loss = d_loss_fn(d_fake_output, torch.FloatTensor(d_real_output.size(0)).uniform_(0.0, 0.3).to(device))

        writer.add_scalar("Discriminator/on_step_real_loss", d_real_loss.mean().item(), num_step)
        writer.add_scalar("Discriminator/on_step_fake_loss", d_fake_loss.mean().item(), num_step)

        d_loss = d_real_loss + d_fake_loss

        d_loss.backward()
        d_optimizer.step()

        if batch_idx % 100 == 0:
            x_grid = make_grid(unnormalize_batch(x_train), nrow=16, padding=2)
            y_grid = make_grid(unnormalize_batch(y_train), nrow=16, padding=2)
            # local_grid = make_grid(unnormalize_batch(x_local), nrow=16, padding=2)
            output_grid = make_grid(torch.clamp(unnormalize_batch(output), min=0.0, max=1.0), nrow=16, padding=2)
            composite_grid = make_grid(torch.clamp(unnormalize_batch(composite), min=0.0, max=1.0), nrow=16, padding=2)
            r_output_grid = make_grid(torch.clamp(unnormalize_batch(r_output), min=0.0, max=1.0), nrow=16, padding=2)
            r_composite_grid = make_grid(torch.clamp(unnormalize_batch(r_composite), min=0.0, max=1.0), nrow=16, padding=2)

            writer.add_image("x_train/epoch_{}".format(epoch), x_grid, num_step)
            writer.add_image("org/epoch_{}".format(epoch), y_grid, num_step)
            # writer.add_image("local/epoch_{}".format(epoch), local_grid, num_step)
            writer.add_image("output/epoch_{}".format(epoch), output_grid, num_step)
            writer.add_image("composite/epoch_{}".format(epoch), composite_grid, num_step)
            writer.add_image("refine_output/epoch_{}".format(epoch), r_output_grid, num_step)
            writer.add_image("refine_composite/epoch_{}".format(epoch), r_composite_grid, num_step)

            print("Step:{}  ".format(num_step),
                  "Epoch:{}".format(epoch),
                  "[{}/{} ".format(batch_idx * len(x_train), len(train_img_loader.dataset)),
                  "({}%)]  ".format(int(100 * batch_idx / float(len(train_img_loader))))
                  )


def test(img_loader, epoch=0):
    l1_criterion = nn.L1Loss()
    l2_criterion = nn.MSELoss()
    ssim_loss = SSIM(window_size=11)
    avg_l1, avg_l2, avg_ssim, avg_psnr = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch_idx, (y_train, x_desc, x_train, x_mask) in tqdm(enumerate(img_loader), ncols=50, desc="Testing",
                                                                  bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
            x_train = x_train.float().to(device)
            x_desc = x_desc.long().to(device)
            x_mask = x_mask.float().to(device)
            y_train = y_train.float().to(device)

            noise = torch.zeros((x_mask.size(0), 256), dtype=torch.float32).normal_().to(device)
            output = net(x_train, x_mask, x_desc, noise)

            l1 = l1_criterion(y_train, output)
            avg_l1 += l1
            mse = l2_criterion(output, y_train)
            avg_l2 += mse
            ssim = ssim_loss(output, y_train)
            avg_ssim += ssim
            psnr = 10.0 * log10(1.0 / mse.item())
            avg_psnr += psnr

        avg_l1 = avg_l1 / len(img_loader)
        avg_l2 = avg_l2 / len(img_loader)
        avg_ssim = avg_ssim / len(img_loader)
        avg_psnr = avg_psnr / len(img_loader)

        writer.add_text("l1", str(avg_l1), epoch)
        writer.add_text("l2", str(avg_l2), epoch)
        writer.add_text("ssim", str(avg_ssim), epoch)
        writer.add_text("psnr", str(avg_psnr), epoch)

        print("L1 Loss:{:.4f}  ".format(avg_l1),
              "L2 Loss:{:.4f}  ".format(avg_l2),
              "SSIM:{:.4f}  ".format(avg_ssim),
              "PSNR:{:.4f}  ".format(avg_psnr))


if __name__ == '__main__':
    if TEST:
        state_dict = torch.load("./runs/Feb01_03-04-30_in-us/weights/weights_with_in_epoch_4.pth")
        net.load_state_dict(state_dict)
        writer.add_text("model", "Feb01_03-04-30_in-us")
        with torch.no_grad():
            for batch_idx, (y_train, x_desc, x_train, x_mask) in tqdm(enumerate(val_img_loader), ncols=50, desc="Testing",
                                                                      bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
                x_train = x_train.float().to(device)
                x_desc = x_desc.long().to(device)
                x_mask = x_mask.float().to(device)
                y_train = y_train.float().to(device)

                noise = torch.zeros((x_mask.size(0), 256), dtype=torch.float32).normal_().to(device)
                output = net(x_train, x_mask, x_desc, noise)
                output = torch.clamp(unnormalize_batch(output), min=0.0, max=1.0)
                x_train = unnormalize_batch(x_train)
                y_train = unnormalize_batch(y_train)
                for i, (x, y, out) in enumerate(zip(x_train, y_train, output)):
                    writer.add_image("x_train/x_train_{}".format(batch_idx), x, i)
                    writer.add_image("y_train/y_train_{}".format(batch_idx), y, i)
                    writer.add_image("output/output_{}".format(batch_idx), out, i)
    else:
        if not os.path.exists("./weights"):
            os.mkdir("./weights")
        writer.add_text("model_name", "IN + US + SA + CL")
        for e in range(NUM_EPOCHS):
            train(e, train_img_loader, train_mask_loader)
            scheduler.step(e)
            r_scheduler.step(e)
            d_scheduler.step(e)
            torch.save(net.state_dict(), "./weights/weights_with_in_wout_desc_epoch_{}.pth".format(e))
            test(val_img_loader, epoch=e)
        writer.close()
