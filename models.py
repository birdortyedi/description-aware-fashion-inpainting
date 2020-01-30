import torch
import torch.nn.functional as F
from torch import nn

from torchvision import transforms
from torchvision import models

from collections import namedtuple
from utils import HDF5Dataset, CentralErasing
from layers import PartialConv2d, LSTMModule, SelfAttention, GaussianNoise


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_0 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(num_features=32)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(num_features=64)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_3 = nn.BatchNorm2d(num_features=128)
        self.conv_4 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        # self.bn_4 = nn.BatchNorm2d(num_features=256)
        # self.conv_5 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=2, padding=1, bias=False)
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x = F.leaky_relu(self.conv_0(x), negative_slope=0.2)
        x = F.leaky_relu(self.bn_1(self.conv_1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn_2(self.conv_2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn_3(self.conv_3(x)), negative_slope=0.2)
        # x = F.leaky_relu(self.bn_4(self.conv_4(x)), negative_slope=0.2)
        x = torch.sigmoid(self.pooling(self.conv_4(x).squeeze()))
        return x


class Net(nn.Module):
    def __init__(self, vocab_size=10000, i_norm=True, attention=True, dilation=True, lstm=True, noise=True):
        super(Net, self).__init__()
        self.attention = attention
        self.dilation = dilation
        self.lstm = lstm
        self.noise = noise
        self.normalization_layer = nn.InstanceNorm2d if i_norm else nn.BatchNorm2d

        self.block_0 = PartialConv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3, multi_channel=True, return_mask=True)
        self.block_1 = PartialConv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, multi_channel=True, return_mask=True)
        self.block_2 = PartialConv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, multi_channel=True, return_mask=True)
        self.block_3 = PartialConv2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, multi_channel=True, return_mask=True)
        self.block_4 = PartialConv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, multi_channel=True, return_mask=True)
        self.dilated_block_1 = PartialConv2d(in_channels=128, out_channels=128, kernel_size=3, padding=2, dilation=2, multi_channel=True)
        self.dilated_block_2 = PartialConv2d(in_channels=128, out_channels=128, kernel_size=3, padding=4, dilation=4, multi_channel=True)
        self.dilated_block_3 = PartialConv2d(in_channels=128, out_channels=128, kernel_size=3, padding=8, dilation=8, multi_channel=True)
        self.dilated_block_4 = PartialConv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, dilation=1, multi_channel=True)
        self.block_6 = PartialConv2d(in_channels=144, out_channels=128, kernel_size=3, padding=1, multi_channel=True, return_mask=True)
        self.block_7 = PartialConv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1, multi_channel=True, return_mask=True)
        self.block_8 = PartialConv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, multi_channel=True, return_mask=True)
        self.block_9 = PartialConv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1, multi_channel=True, return_mask=True)
        self.block_10 = PartialConv2d(in_channels=160, out_channels=64, kernel_size=3, padding=1, multi_channel=True, return_mask=True)
        self.block_11 = PartialConv2d(in_channels=67, out_channels=3, kernel_size=3, padding=1, multi_channel=True, return_mask=True)

        self.norm_1 = self.normalization_layer(num_features=64)
        self.norm_2 = self.normalization_layer(num_features=128)
        self.norm_3 = self.normalization_layer(num_features=64)
        self.norm_4 = self.normalization_layer(num_features=128)
        self.dilated_norm_1 = self.normalization_layer(num_features=128)
        self.dilated_norm_2 = self.normalization_layer(num_features=128)
        self.dilated_norm_3 = self.normalization_layer(num_features=128)
        self.dilated_norm_4 = self.normalization_layer(num_features=128)
        self.norm_6 = self.normalization_layer(num_features=128)
        self.norm_7 = self.normalization_layer(num_features=128)
        self.norm_8 = self.normalization_layer(num_features=128)
        self.norm_9 = self.normalization_layer(num_features=128)
        self.norm_10 = self.normalization_layer(num_features=64)
        self.s_attention_8 = SelfAttention(in_channels=128)
        self.s_attention_9 = SelfAttention(in_channels=128)

        if self.lstm:
            self.block_5 = PartialConv2d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2, multi_channel=True, return_mask=True)
            self.norm_5 = self.normalization_layer(num_features=128)
        else:
            self.block_5 = PartialConv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, multi_channel=True, return_mask=True)
            self.norm_5 = self.normalization_layer(num_features=256)

        self.lstm_block = LSTMModule(vocab_size, embedding_dim=32, hidden_dim=1024, n_layers=3, output_size=128)
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.noise_layer = GaussianNoise()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2.0)
        self.conv = nn.Conv2d(in_channels=128, out_channels=16, kernel_size=1)

    def forward(self, x, mask=None, descriptions=None, noise=None):
        x_0, m_0 = self.block_0(x, mask)
        x_0 = F.relu(x_0)
        x_1, m_1 = self.block_1(x_0, m_0)
        x_1 = F.relu(self.norm_1(x_1))
        x_2, m_2 = self.block_2(x_1, m_1)
        x_2 = F.relu(self.norm_2(x_2))
        x_3, m_3 = self.block_3(x_2, m_2)
        x_3 = F.relu(self.norm_3(x_3))
        if self.dilation:
            dilated_x, dilated_m = self.dilated_block_1(x_3, m_3)
            dilated_x = F.relu(self.dilated_norm_1(dilated_x))
            dilated_x, dilated_m = self.dilated_block_2(dilated_x, dilated_m)
            dilated_x = F.relu(self.dilated_norm_2(dilated_x))
            dilated_x, dilated_m = self.dilated_block_3(dilated_x, dilated_m)
            dilated_x = F.relu(self.dilated_norm_3(dilated_x))
            dilated_x, dilated_m = self.dilated_block_4(dilated_x, dilated_m)
            x_3 = F.relu(self.dilated_norm_4(dilated_x))
        x_4, m_4 = self.block_4(x_3, m_3)
        x_4 = F.relu(self.norm_4(x_4))
        x_5, m_5 = self.block_5(x_4, m_4)
        x_5 = F.relu(self.norm_5(x_5))

        visual_embedding = self.pooling(x_5).squeeze()
        if self.lstm:
            textual_embedding = self.lstm_block(descriptions)
            embedding = torch.cat((visual_embedding, textual_embedding), dim=1)
            if self.noise:
                embedding = self.noise_layer(embedding, noise)
            out = embedding.view(-1, 16, 4, 4)
        else:
            if self.noise:
                visual_embedding = self.noise_layer(visual_embedding, noise)
            out = visual_embedding.view(-1, 16, 4, 4)

        x_6 = self.upsample(out)
        x_6 = torch.cat((x_4, x_6), dim=1)
        m_6 = self.conv(m_4)
        m_6 = torch.where(m_6 > 0.5, torch.ones_like(m_6).float(), torch.zeros_like(m_6).float())
        m_6 = torch.cat((m_4, m_6), dim=1)
        x_6, m_6 = self.block_6(x_6, m_6)
        out = F.leaky_relu(self.norm_6(x_6), negative_slope=0.2)

        x_7 = self.upsample(out)
        x_7 = torch.cat((x_3, x_7), dim=1)
        m_7 = self.upsample(m_6)
        m_7 = torch.cat((m_3, m_7), dim=1)
        x_7, m_7 = self.block_7(x_7, m_7)
        out = F.leaky_relu(self.norm_7(x_7), negative_slope=0.2)

        x_8 = self.upsample(out)
        x_8 = torch.cat((x_2, x_8), dim=1)
        m_8 = self.upsample(m_7)
        m_8 = torch.cat((m_2, m_8), dim=1)
        x_8, m_8 = self.block_8(x_8, m_8)
        out = F.leaky_relu(self.norm_8(x_8), negative_slope=0.2)
        if self.attention:
            out, x_8_attention = self.s_attention_8(out)
            out = F.relu(out)

        x_9 = self.upsample(out)
        x_9 = torch.cat((x_1, x_9), dim=1)
        m_9 = self.upsample(m_8)
        m_9 = torch.cat((m_1, m_9), dim=1)
        x_9, m_9 = self.block_9(x_9, m_9)
        out = F.leaky_relu(self.norm_9(x_9), negative_slope=0.2)
        if self.attention:
            out, x_9_attention = self.s_attention_9(out)
            out = F.relu(out)

        x_10 = self.upsample(out)
        x_10 = torch.cat((x_0, x_10), dim=1)
        m_10 = self.upsample(m_9)
        m_10 = torch.cat((m_0, m_10), dim=1)
        x_10, m_10 = self.block_10(x_10, m_10)
        out = F.leaky_relu(self.norm_10(x_10), negative_slope=0.2)

        x_11 = self.upsample(out)
        x_11 = torch.cat((x, x_11), dim=1)
        m_11 = self.upsample(m_10)
        m_11 = torch.cat((mask, m_11), dim=1)
        x_11, _ = self.block_11(x_11, m_11)
        out = torch.tanh(x_11)

        return out


class BaseNet(nn.Module):
    def __init__(self, vocab_size=10000, i_norm=True, attention=True, dilation=True, lstm=True, noise=True):
        self.attention = attention
        self.dilation = dilation
        self.lstm = lstm
        self.noise = noise
        self.normalization_layer = nn.InstanceNorm2d if i_norm else nn.BatchNorm2d

        self.block_0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3)
        self.block_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.block_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.block_3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.block_4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.dilated_block_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=2, dilation=2)
        self.dilated_block_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=4, dilation=4)
        self.dilated_block_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=8, dilation=8)
        self.dilated_block_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, dilation=1)
        self.block_6 = nn.Conv2d(in_channels=144, out_channels=128, kernel_size=3, padding=1)
        self.block_7 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1)
        self.block_8 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.block_9 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1)
        self.block_10 = nn.Conv2d(in_channels=160, out_channels=64, kernel_size=3, padding=1)
        self.block_11 = nn.Conv2d(in_channels=67, out_channels=3, kernel_size=3, padding=1)

        self.norm_1 = self.normalization_layer(num_features=64)
        self.norm_2 = self.normalization_layer(num_features=128)
        self.norm_3 = self.normalization_layer(num_features=64)
        self.norm_4 = self.normalization_layer(num_features=128)
        self.dilated_norm_1 = self.normalization_layer(num_features=128)
        self.dilated_norm_2 = self.normalization_layer(num_features=128)
        self.dilated_norm_3 = self.normalization_layer(num_features=128)
        self.dilated_norm_4 = self.normalization_layer(num_features=128)
        self.norm_6 = self.normalization_layer(num_features=128)
        self.norm_7 = self.normalization_layer(num_features=128)
        self.norm_8 = self.normalization_layer(num_features=128)
        self.norm_9 = self.normalization_layer(num_features=128)
        self.norm_10 = self.normalization_layer(num_features=64)
        self.s_attention_8 = SelfAttention(in_channels=128)
        self.s_attention_9 = SelfAttention(in_channels=128)

        if self.lstm:
            self.block_5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=2,  padding=2)
            self.norm_5 = self.normalization_layer(num_features=128)
        else:
            self.block_5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2)
            self.norm_5 = self.normalization_layer(num_features=256)

        self.lstm_block = LSTMModule(vocab_size, embedding_dim=32, hidden_dim=1024, n_layers=3, output_size=128)
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.noise_layer = GaussianNoise()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2.0)
        self.conv = nn.Conv2d(in_channels=128, out_channels=16, kernel_size=1)

    def forward(self, x, mask=None, descriptions=None, noise=None):
        x_0 = F.relu(self.block_0(x))
        x_1 = F.relu(self.norm_1(self.block_1(x_0)))
        x_2 = F.relu(self.norm_2(self.block_2(x_1)))
        x_3 = F.relu(self.norm_3(self.block_3(x_2)))
        if self.dilation:
            dilated_x = F.relu(self.dilated_norm_1(self.dilated_block_1(x_3)))
            dilated_x = F.relu(self.dilated_norm_2(self.dilated_block_2(dilated_x)))
            dilated_x = F.relu(self.dilated_norm_3(self.dilated_block_3(dilated_x)))
            x_3 = F.relu(self.dilated_norm_4(self.dilated_block_4(dilated_x)))
        x_4 = F.relu(self.norm_4(self.block_4(x_3)))
        x_5 = F.relu(self.norm_5(self.block_5(x_4)))

        visual_embedding = self.pooling(x_5).squeeze()
        if self.lstm:
            textual_embedding = self.lstm_block(descriptions)
            embedding = torch.cat((visual_embedding, textual_embedding), dim=1)
            if self.noise:
                embedding = self.noise_layer(embedding, noise)
            out = embedding.view(-1, 16, 4, 4)
        else:
            if self.noise:
                visual_embedding = self.noise_layer(visual_embedding, noise)
            out = visual_embedding.view(-1, 16, 4, 4)

        x_6 = self.upsample(out)
        x_6 = torch.cat((x_4, x_6), dim=1)
        out = F.leaky_relu(self.norm_6(self.block_6(x_6)), negative_slope=0.2)

        x_7 = self.upsample(out)
        x_7 = torch.cat((x_3, x_7), dim=1)
        out = F.leaky_relu(self.norm_7(self.block_7(x_7)), negative_slope=0.2)

        x_8 = self.upsample(out)
        x_8 = torch.cat((x_2, x_8), dim=1)
        out = F.leaky_relu(self.norm_8(self.block_8(x_8)), negative_slope=0.2)
        if self.attention:
            out, x_8_attention = self.s_attention_8(out)
            out = F.relu(out)

        x_9 = self.upsample(out)
        x_9 = torch.cat((x_1, x_9), dim=1)
        out = F.leaky_relu(self.norm_9(self.block_9(x_9)), negative_slope=0.2)
        if self.attention:
            out, x_9_attention = self.s_attention_9(out)
            out = F.relu(out)

        x_10 = self.upsample(out)
        x_10 = torch.cat((x_0, x_10), dim=1)
        out = F.leaky_relu(self.norm_10(self.block_10(x_10)), negative_slope=0.2)

        x_11 = self.upsample(out)
        x_11 = torch.cat((x, x_11), dim=1)
        out = torch.tanh(self.block_11(x_11))

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
