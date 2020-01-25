import torch
from torch import nn, optim
from torch.utils import data
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from tqdm import tqdm
from colorama import Fore

from utils import HDF5Dataset, weights_init, normalize_batch, unnormalize_batch
from models import Net, Discriminator, VGG16
from losses import CustomLoss

NUM_EPOCHS = 10
BATCH_SIZE = 16

fg_train = HDF5Dataset(filename='./Fashion-Gen/fashiongen_256_256_train.h5')
fg_val = HDF5Dataset(filename='./Fashion-Gen/fashiongen_256_256_validation.h5', is_train=False)

print("Sample size in training: {}".format(len(fg_train)))

train_loader = data.DataLoader(fg_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = data.DataLoader(fg_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d_net = Discriminator()
net = Net(vocab_size=fg_train.vocab_size)
vgg = VGG16(requires_grad=False)
if torch.cuda.device_count() > 1:
    print("Using {} GPUs...".format(torch.cuda.device_count()))
    d_net = nn.DataParallel(d_net).to(device)
    net = nn.DataParallel(net).to(device)
vgg.to(device)

net.apply(weights_init)

d_loss_fn = nn.BCELoss()
d_loss_fn = d_loss_fn.to(device)
loss_fn = CustomLoss()
loss_fn = loss_fn.to(device)

lr, d_lr = 0.0002, 0.0001
d_optimizer = optim.Adam(d_net.parameters(), lr=d_lr, betas=(0.9, 0.999))
optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.0, 0.999))

d_scheduler = optim.lr_scheduler.ExponentialLR(d_optimizer, gamma=0.9)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

writer = SummaryWriter()


def train(epoch, loader):
    for batch_idx, (x_train, x_desc, x_mask, x_local, y_train) in tqdm(enumerate(loader), ncols=50, desc="Training",
                                                                       bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
        num_step = epoch * len(loader) + batch_idx
        x_train = x_train.float().to(device)
        x_desc = x_desc.long().to(device)
        x_mask = x_mask.float().to(device)
        x_local = x_local.float().to(device)
        y_train = y_train.float().to(device)

        net.zero_grad()
        output = net(x_train, x_desc, x_mask)
        d_output = d_net(output.detach())
        composite = x_mask * y_train + (1.0 - x_mask) * output

        vgg_features_gt = vgg(normalize_batch(unnormalize_batch(y_train)))
        vgg_features_composite = vgg(composite)
        vgg_features_output = vgg(output)

        total_loss, pixel_valid_loss, pixel_hole_loss,\
            content_loss, style_loss, tv_loss, adversarial_loss = loss_fn(y_train, output, composite, x_mask, d_output,
                                                                          vgg_features_gt, vgg_features_composite, vgg_features_output)

        writer.add_scalar("Generator/on_step_total_loss", total_loss.item(), num_step)
        writer.add_scalar("Generator/on_step_pixel_valid_loss", pixel_valid_loss.item(), num_step)
        writer.add_scalar("Generator/on_step_pixel_hole_loss", pixel_hole_loss.item(), num_step)
        writer.add_scalar("Generator/on_step_content_loss", content_loss.item(), num_step)
        writer.add_scalar("Generator/on_step_style_loss", style_loss.item(), num_step)
        writer.add_scalar("Generator/on_step_tv_loss", tv_loss.item(), num_step)
        writer.add_scalar("Generator/on_step_adversarial_loss", adversarial_loss.item(), num_step)
        writer.add_scalar("LR/learning_rate", scheduler.get_lr(), num_step)

        total_loss.backward()
        optimizer.step()

        d_net.zero_grad()
        d_real_output = d_net(y_train).view(-1)
        d_fake_output = d_net(net(x_train, x_desc, x_mask)).view(-1)

        if torch.rand(1) > 0.2:
            d_real_loss = d_loss_fn(d_real_output, torch.FloatTensor(d_real_output.size(0)).uniform_(0.0, 0.3))
            d_fake_loss = d_loss_fn(d_fake_output, torch.FloatTensor(d_fake_output.size(0)).uniform_(0.7, 1.2))
        else:
            d_real_loss = d_loss_fn(d_real_output, torch.FloatTensor(d_fake_output.size(0)).uniform_(0.7, 1.2))
            d_fake_loss = d_loss_fn(d_fake_output, torch.FloatTensor(d_real_output.size(0)).uniform_(0.0, 0.3))

        writer.add_scalar("Discriminator/on_step_real_loss", d_real_loss.mean().item(), num_step)
        writer.add_scalar("Discriminator/on_step_fake_loss", d_fake_loss.mean().item(), num_step)

        d_loss = d_real_loss + d_fake_loss

        d_loss.backward()
        d_optimizer.step()

        if batch_idx % 100 == 0:
            x_grid = make_grid(unnormalize_batch(x_train), nrow=16, padding=2)
            y_grid = make_grid(unnormalize_batch(y_train), nrow=16, padding=2)
            local_grid = make_grid(unnormalize_batch(x_local), nrow=16, padding=2)
            output_grid = make_grid(torch.clamp(unnormalize_batch(output), min=0.0, max=1.0), nrow=16, padding=2)
            composite_grid = make_grid(torch.clamp(unnormalize_batch(composite), min=0.0, max=1.0), nrow=16, padding=2)
            writer.add_image("x_train/epoch_{}".format(epoch), x_grid, num_step)
            writer.add_image("org/epoch_{}".format(epoch), y_grid, num_step)
            writer.add_image("local/epoch_{}".format(epoch), local_grid, num_step)
            writer.add_image("output/epoch_{}".format(epoch), output_grid, num_step)
            writer.add_image("composite/epoch_{}".format(epoch), composite_grid, num_step)

            print("Step:{}  ".format(num_step),
                  "Epoch:{}".format(epoch),
                  "[{}/{} ".format(batch_idx * len(x_train), len(train_loader.dataset)),
                  "({}%)]  ".format(int(100 * batch_idx / float(len(train_loader)))),
                  "Loss: {:.6f} ".format(total_loss.item()),
                  "Valid: {:.6f} ".format(pixel_valid_loss.item()),
                  "Hole: {:.6f} ".format(pixel_hole_loss.item()),
                  "Content: {:.5f} ".format(content_loss.item()),
                  "Style: {:.6f} ".format(style_loss.item()),
                  "TV: {:.6f} ".format(tv_loss.item()),
                  "Adv.: {:.6f} ".format(adversarial_loss.item())
                  )


if __name__ == '__main__':
    for e in range(NUM_EPOCHS):
        train(e, train_loader)
        scheduler.step(e)
        d_scheduler.step(e)
        torch.save(net.state_dict(), "./weights/weights_epoch_{}.pth".format(e))
    writer.close()
