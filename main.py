import torch
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from tensorboardX import SummaryWriter

from tqdm import tqdm
from colorama import Fore

from utils import HDF5Dataset, RandomCentralErasing, UnNormalize
from models import Net, AdvancedNet
from losses import CustomInpaintingLoss

NUM_EPOCHS = 250
BATCH_SIZE = 128


train_transform = transforms.Compose([transforms.ToTensor(),
                                      RandomCentralErasing(p=1.0, scale=(0.03, 0.12), ratio=(0.75, 1.25)),
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
net = AdvancedNet(fg_train.vocab_size)  # Net(fg_train.vocab_size)
if torch.cuda.device_count() > 1:
    print("Using {} GPUs...".format(torch.cuda.device_count()))
    net = nn.DataParallel(net)
net.to(device)

loss_fn = CustomInpaintingLoss()
loss_fn = loss_fn.to(device)
lr = 0.01
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
writer = SummaryWriter()


def train(epoch, loader, l_fn, opt, sch):
    net.train()
    total_loss, total_content_loss, total_style_loss, total_struct_loss, total_adverserial_loss = 0., 0., 0., 0., 0.
    for batch_idx, (x_train, x_desc, y_train) in tqdm(enumerate(loader), ncols=50, desc="Training",
                                                      bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
        opt.zero_grad()
        x_train = x_train.float().to(device)
        x_desc = x_desc.long().to(device)
        y_train = y_train.float().to(device)

        output, d_x, d_out = net(x_train, x_desc, y_train)
        loss, content, style, struct, adversarial = l_fn(output, y_train,  d_x, d_out)

        total_loss += loss.item()
        total_content_loss += content.item()
        total_style_loss += style.item()
        total_struct_loss += struct.item()
        total_adverserial_loss += adversarial.item()

        loss.backward()
        opt.step()
        sch.step(epoch)

        writer.add_scalar("Loss/on_step_loss", loss.item(), epoch * len(loader) + batch_idx)
        writer.add_scalar("Loss/on_step_content_loss", content.item(), epoch * len(loader) + batch_idx)
        writer.add_scalar("Loss/on_step_style_loss", style.item(), epoch * len(loader) + batch_idx)
        writer.add_scalar("Loss/on_step_structure_loss", struct.item(), epoch * len(loader) + batch_idx)
        writer.add_scalar("Loss/on_step_adversarial_loss", adversarial.item(), epoch * len(loader) + batch_idx)

        if batch_idx % 100 == 0:
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

    writer.add_scalar("Loss/on_epoch_loss", total_loss, epoch)
    writer.add_scalar("Loss/on_epoch_content_loss", total_content_loss, epoch)
    writer.add_scalar("Loss/on_epoch_style_loss", total_style_loss, epoch)
    writer.add_scalar("Loss/on_epoch_structure_loss", total_struct_loss, epoch)
    writer.add_scalar("Loss/on_epoch_adverserial_loss", total_adverserial_loss, epoch)


def evaluate(epoch, loader, l_fn):
    total_loss, total_content_loss, total_style_loss, total_struct_loss, total_adverserial_loss = 0., 0., 0., 0., 0.
    with torch.no_grad():
        net.eval()
        for batch_idx, (x_val, x_desc_val, y_val) in tqdm(enumerate(loader), ncols=50, desc="Validation",
                                              bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
            x_val = x_val.float().to(device)
            x_desc_val = x_desc_val.long().to(device)
            y_val = y_val.float().to(device)

            output, d_x, d_out = net(x_val, x_desc_val, y_val)
            val_loss, val_content, val_style, val_struct, val_adversarial = l_fn(output, y_val, d_x, d_out)

            total_loss += val_loss.item()
            total_content_loss += val_content.item()
            total_style_loss += val_style.item()
            total_struct_loss += val_struct.item()
            total_adverserial_loss += val_adversarial.item()

            if batch_idx % 100 == 0:
                print("[{}/{} ".format(batch_idx * len(x_val), len(loader.dataset)),
                      "({}%)]\t".format(int(100 * batch_idx / float(len(loader)))),
                      "Loss: {:.4f}".format(val_loss.item()),
                      "Content: {:.4f}  ".format(val_content.item()),
                      "Style: {:.9f}  ".format(val_style.item()),
                      "Structure: {:.4f}  ".format(val_struct.item()),
                      "Adversarial: {:.4f}  ".format(val_adversarial.item()))

        writer.add_scalar("Loss/on_epoch_val_loss", total_loss, epoch)
        writer.add_scalar("Loss/on_epoch_val_content_loss", total_content_loss, epoch)
        writer.add_scalar("Loss/on_epoch_val_style_loss", total_style_loss, epoch)
        writer.add_scalar("Loss/on_epoch_val_structure_loss", total_struct_loss, epoch)
        writer.add_scalar("Loss/on_epoch_val_adversarial_loss", total_adverserial_loss, epoch)


if __name__ == '__main__':
    for e in range(NUM_EPOCHS):
        train(e, train_loader, loss_fn, optimizer, scheduler)
        evaluate(e, val_loader, loss_fn)
        torch.save(net.state_dict(), "./weights/weights_epoch_{}.pth".format(e))
