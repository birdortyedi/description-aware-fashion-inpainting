import torch
import torch.nn.functional as F
from torch import nn

from torchvision import transforms
from torchvision import models

from collections import namedtuple
from utils import HDF5Dataset, CentralErasing
from layers import PartialConv2d
from blocks import lstm_block, dilated_res_blocks, SelfAttention, DilatedResidualBlock, conv_bn_lrelu_block, conv_lrelu_block


class LocalDiscriminator(nn.Module):
    def __init__(self):
        super(LocalDiscriminator, self).__init__()
        self.d_block_1 = conv_lrelu_block(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.d_block_2 = conv_bn_lrelu_block(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.d_block_3 = conv_bn_lrelu_block(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.d_block_4 = conv_bn_lrelu_block(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.d_block_5 = conv_bn_lrelu_block(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.d_block_6 = conv_bn_lrelu_block(in_channels=192, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout2d(p=0.3)
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.d_block_7 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = self.d_block_1(x)

        x_1 = self.dropout(self.d_block_2(x))
        x_2 = self.dropout(self.d_block_3(x_1))
        x = torch.cat((x_1, x_2), dim=1)

        x_1 = self.dropout(self.d_block_4(x))
        x_2 = self.dropout(self.d_block_5(x_1))
        x = torch.cat((x_1, x_2), dim=1)

        x = self.d_block_6(x)
        x = self.avg_pooling(x).squeeze()
        x = torch.sigmoid(self.d_block_7(x))
        return x


class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super(GlobalDiscriminator, self).__init__()
        # Discriminator
        self.d_block_1 = conv_lrelu_block(in_channels=3, out_channels=16, kernel_size=7, stride=2, padding=1)
        self.d_block_2 = conv_bn_lrelu_block(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.d_block_3 = conv_bn_lrelu_block(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.d_block_4 = conv_bn_lrelu_block(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=1)
        self.d_block_5 = conv_bn_lrelu_block(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.d_block_6 = conv_bn_lrelu_block(in_channels=64, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout2d(p=0.3)
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.d_block_7 = nn.Linear(in_features=16, out_features=1)

    def forward(self, x):
        x = self.d_block_1(x)
        x = self.dropout(self.d_block_2(x))
        x = self.dropout(self.d_block_3(x))
        x = self.dropout(self.d_block_4(x))
        x = self.dropout(self.d_block_5(x))
        x = self.d_block_6(x)
        x = self.avg_pooling(x)
        x = torch.sigmoid(self.d_block_7(x.squeeze()))
        return x


class RefineNet_old(nn.Module):
    def __init__(self):
        super(RefineNet_old, self).__init__()

        # Encoder c'ing
        self.block_5 = conv_in_lrelu_block(in_channels=64, out_channels=256, kernel_size=5, stride=2, padding=2)

        # Decoder
        self.block_6 = deconv_in_lrelu_block(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)
        self._1x1conv_6 = one_by_one_conv_lrelu_block(in_channels=288, out_channels=64)
        self.block_7 = deconv_in_lrelu_block(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x_1 = self.block_1(x)
        x_2 = self.dropout(self.block_2(x_1))
        x_3 = self.dropout(self.block_3(x_2))

        dil_res_x_3 = self.dilated_res_blocks(x_3)
        attention_map, _ = self.self_attention(dil_res_x_3)

        x_4 = self.dropout(self.block_4(dil_res_x_3))
        x_5 = self.dropout(self.block_5(x_4))

        visual_embedding = self.avg_pooling(x_5).squeeze()

        x_6 = self.block_6(visual_embedding.view(-1, 16, 4, 4))
        x_6 = torch.cat((x_5, x_6), dim=1)
        x_6 = self.dropout(self._1x1conv_6(x_6))

        x_7 = self.block_7(x_6)
        x_7 = self.dropout(torch.cat((x_4, x_7), dim=1))

        x_8 = self.block_8(x_7)
        x_8 = torch.cat((x_3, x_8, attention_map), dim=1)
        x_8 = self.dropout(self._1x1conv_8(x_8))

        x_9 = self.block_9(x_8)
        x_9 = torch.cat((x_2, x_9), dim=1)
        x_9 = self.dropout(self._1x1conv_9(x_9))

        x_10 = self.block_10(x_9)
        x_10 = torch.cat((x_1, x_10), dim=1)

        x_11 = self.block_11(x_10)

        return x_11


class Net(nn.Module):
    def __init__(self, vocab_size):
        super(Net, self).__init__()
        self.block_0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3)
        self.block_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.in_1 = nn.InstanceNorm2d(num_features=64)
        self.s_attention_1 = SelfAttention(in_channels=64)
        self.block_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.in_2 = nn.InstanceNorm2d(num_features=128)
        self.s_attention_2 = SelfAttention(in_channels=128)
        self.block_3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.in_3 = nn.InstanceNorm2d(num_features=64)
        self.s_attention_3 = SelfAttention(in_channels=64)
        self.block_4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.in_4 = nn.InstanceNorm2d(num_features=128)
        self.s_attention_4 = SelfAttention(in_channels=128)
        self.block_5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.in_5 = nn.InstanceNorm2d(num_features=128)
        self.s_attention_5 = SelfAttention(in_channels=128)

        self.lstm_block = lstm_block(vocab_size, output_size=128)
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.upsample = nn.Upsample(mode="nearest", scale_factor=2.0)

        self.block_6 = nn.Conv2d(in_channels=144, out_channels=128, kernel_size=3, padding=1)
        self.in_6 = nn.InstanceNorm2d(num_features=128)
        self.block_7 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1)
        self.in_7 = nn.InstanceNorm2d(num_features=128)
        self.block_8 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.in_8 = nn.InstanceNorm2d(num_features=128)
        self.block_9 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1)
        self.in_9 = nn.InstanceNorm2d(num_features=128)
        self.block_10 = nn.Conv2d(in_channels=160, out_channels=64, kernel_size=3, padding=1)
        self.in_10 = nn.InstanceNorm2d(num_features=64)
        self.block_11 = nn.Conv2d(in_channels=67, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x, descriptions):
        x_0 = F.relu(self.block_0(x))
        x_1 = F.relu(self.in_1(self.block_1(x_0)))
        x_1, _ = self.s_attention_1(x_1)
        x_1 = F.relu(x_1)
        x_2 = F.relu(self.in_2(self.block_2(x_1)))
        x_2, _ = self.s_attention_2(x_2)
        x_2 = F.relu(x_2)
        x_3 = F.relu(self.in_3(self.block_3(x_2)))
        x_3, _ = self.s_attention_3(x_3)
        x_3 = F.relu(x_3)
        x_4 = F.relu(self.in_4(self.block_4(x_3)))
        x_4, _ = self.s_attention_4(x_4)
        x_4 = F.relu(x_4)
        x_5 = F.relu(self.in_5(self.block_5(x_4)))
        x_5, _ = self.s_attention_5(x_5)
        x_5 = F.relu(x_5)

        visual_embedding = self.pooling(x_5).squeeze()
        textual_embedding = self.lstm_block(descriptions)
        embedding = torch.cat((visual_embedding, textual_embedding), dim=1)
        out = embedding.view(-1, 16, 4, 4)

        x_6 = self.upsample(out)
        x_6 = torch.cat((x_4, x_6), dim=1)
        x_6 = self.block_6(x_6)
        out = F.leaky_relu(self.in_6(x_6), negative_slope=0.2)

        x_7 = self.upsample(out)
        x_7 = torch.cat((x_3, x_7), dim=1)
        x_7 = self.block_7(x_7)
        out = F.leaky_relu(self.in_7(x_7), negative_slope=0.2)

        x_8 = self.upsample(out)
        x_8 = torch.cat((x_2, x_8), dim=1)
        x_8 = self.block_8(x_8)
        out = F.leaky_relu(self.in_8(x_8), negative_slope=0.2)

        x_9 = self.upsample(out)
        x_9 = torch.cat((x_1, x_9), dim=1)
        x_9 = self.block_9(x_9)
        out = F.leaky_relu(self.in_9(x_9), negative_slope=0.2)

        x_10 = self.upsample(out)
        x_10 = torch.cat((x_0, x_10), dim=1)
        x_10 = self.block_10(x_10)
        out = F.leaky_relu(self.in_10(x_10), negative_slope=0.2)

        out = torch.tanh(self.block_12(self.upsample(out)))

        return out


class CoarseNet(nn.Module):
    def __init__(self, vocab_size):
        super(CoarseNet, self).__init__()

        # Encoder
        self.block_1 = PartialConv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3, return_mask=True, multi_channel=True)

        self.block_2 = PartialConv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, return_mask=True, multi_channel=True)
        self.in_2 = nn.InstanceNorm2d(num_features=64)

        self.block_3 = PartialConv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, return_mask=True, multi_channel=True)
        self.in_3 = nn.InstanceNorm2d(num_features=128)

        # Dilated Residual Blocks
        self.dilated_res_block_1 = DilatedResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=2, padding=2)
        self.dilated_res_block_2 = DilatedResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=2, padding=2)
        self.dilated_res_block_3 = DilatedResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=2, padding=2)

        # Self-Attention
        self.self_attention = SelfAttention(in_channels=128)

        # Visual features for concatenating with textual features
        self.block_4 = PartialConv2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, return_mask=True, multi_channel=True)
        self.in_4 = nn.InstanceNorm2d(num_features=64)

        self.block_5 = PartialConv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, return_mask=True, multi_channel=True)
        self.in_5 = nn.InstanceNorm2d(num_features=128)

        self.block_6 = PartialConv2d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2, return_mask=True, multi_channel=True)
        self.in_6 = nn.InstanceNorm2d(num_features=128)

        # LSTM
        self.lstm_block = lstm_block(vocab_size, output_size=128)

        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout2d(p=0.3)
        self.upsample = nn.Upsample(mode="nearest", scale_factor=2.0)

        # Decoder
        self.block_7 = nn.Conv2d(in_channels=144, out_channels=128, kernel_size=3, padding=1)
        self.in_7 = nn.InstanceNorm2d(num_features=128)

        self.block_8 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1)
        self.in_8 = nn.InstanceNorm2d(num_features=128)

        self.block_9 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.in_9 = nn.InstanceNorm2d(num_features=128)

        self.block_10 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1)
        self.in_10 = nn.InstanceNorm2d(num_features=128)

        self.block_11 = nn.Conv2d(in_channels=160, out_channels=64, kernel_size=3, padding=1)
        self.in_11 = nn.InstanceNorm2d(num_features=64)

        self.block_12 = nn.Conv2d(in_channels=67, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x, descriptions, mask):
        x_1, m_1 = self.block_1(x, mask)
        x_1 = F.relu(x_1)
        x_2, m_2 = self.block_2(x_1, m_1)
        x_2 = F.relu(self.in_2(x_2))
        x_3, m_3 = self.block_3(x_2, m_2)
        x_3 = F.relu(self.in_3(x_3))

        dil_res_x_3_1 = self.dilated_res_block_1(x_3, m_3)
        dil_res_x_3_2 = self.dilated_res_block_2(dil_res_x_3_1, m_3)
        dil_res_x_3_3 = self.dilated_res_block_3(dil_res_x_3_2, m_3)

        att_feat_map, attention_map = self.self_attention(dil_res_x_3_3)

        x_4, m_4 = self.block_4(att_feat_map, m_3)
        x_4 = F.relu(self.in_4(x_4))
        x_5, m_5 = self.block_5(x_4, m_4)
        x_5 = F.relu(self.in_5(x_5))
        x_6, m_6 = self.block_6(x_5, m_5)
        x_6 = F.relu(self.in_6(x_6))

        visual_embedding = self.avg_pooling(x_6).squeeze()
        textual_embedding = self.lstm_block(descriptions)
        embedding = torch.cat((visual_embedding, textual_embedding), dim=1)

        out = self.upsample(embedding.view(-1, 16, 4, 4))
        out = torch.cat((x_5, out), dim=1)
        out = self.block_7(out)
        out = F.leaky_relu(self.in_7(out), negative_slope=0.2)

        out = self.upsample(out)
        out = torch.cat((x_4, out), dim=1)
        out = self.block_8(out)
        out = F.leaky_relu(self.in_8(out), negative_slope=0.2)

        out = self.upsample(out)
        b, c, w, h = out.size()
        out_reshaped = out.view(b, -1, w * h)
        out_reshaped = torch.bmm(out_reshaped, attention_map.permute(0, 2, 1))
        attention_out = out_reshaped.view(b, c, w, h)
        out = out + attention_out
        out = F.leaky_relu(out, negative_slope=0.2)
        out = torch.cat((x_3, out), dim=1)
        out = self.block_9(out)
        out = F.leaky_relu(self.in_9(out), negative_slope=0.2)

        out = self.upsample(out)
        out = torch.cat((x_2, out), dim=1)
        out = self.block_10(out)
        out = F.leaky_relu(self.in_10(out), negative_slope=0.2)

        out = self.upsample(out)
        out = torch.cat((x_1, out), dim=1)
        out = self.block_11(out)
        out = F.leaky_relu(self.in_11(out), negative_slope=0.2)

        out = self.upsample(out)
        out = torch.cat((x, out), dim=1)
        out = torch.tanh(self.block_12(out))

        return out


class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


if __name__ == '__main__':
    train_transform = transforms.Compose([transforms.ToTensor(),
                                          CentralErasing(scale=(0.03, 0.12), ratio=(0.75, 1.25), value=1),
                                          # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                          ])

    fg_train = HDF5Dataset(filename='./Fashion-Gen/fashiongen_256_256_train.h5')
    print("Sample size in training: {}".format(len(fg_train)))

    coarse = CoarseNet(30000)
    refine = RefineNet()
    local = LocalDiscriminator()
    _global = GlobalDiscriminator()
