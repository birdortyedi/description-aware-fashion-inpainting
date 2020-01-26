import torch
import torch.nn.functional as F
from torch import nn

from layers import PartialConv2d


def conv_in_lrelu_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.InstanceNorm2d(num_features=out_channels),
        nn.LeakyReLU(negative_slope=0.2)
    )


def conv_bn_lrelu_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(num_features=out_channels),
        nn.LeakyReLU(negative_slope=0.2)
    )


def conv_lrelu_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.LeakyReLU(negative_slope=0.2)
    )


def dilated_res_blocks(num_features, kernel_size=3, stride=1, dilation=2, padding=2):
    return nn.Sequential(
        DilatedResidualBlock(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding),
        DilatedResidualBlock(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding),
        DilatedResidualBlock(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding),
        DilatedResidualBlock(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding),
    )


def lstm_block(vocab_size, embedding_dim=32, hidden_dim=1024, n_layers=3, output_size=128):
    return nn.Sequential(
        LSTMModule(vocab_size, embedding_dim, hidden_dim, n_layers, output_size)
    )


def upsampling_in_lrelu_block(in_channels, out_channels, mode='nearest', scale_factor=2.0, padding=0):
    return nn.Sequential(
        nn.Upsample(mode=mode, scale_factor=scale_factor),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=padding),
        nn.InstanceNorm2d(num_features=out_channels, affine=True),
        nn.LeakyReLU(negative_slope=0.2)
    )


def deconv_in_lrelu_block(in_channels, out_channels, kernel_size, stride=2, padding=0):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=0),
        nn.InstanceNorm2d(num_features=out_channels, affine=True),
        nn.LeakyReLU(negative_slope=0.2)
    )


def upsampling_bn_lrelu_block(in_channels, out_channels, mode='nearest', scale_factor=2.0, padding=0):
    return nn.Sequential(
        nn.Upsample(mode=mode, scale_factor=scale_factor),
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=padding),
        nn.BatchNorm2d(num_features=out_channels),
        nn.LeakyReLU(negative_slope=0.2)
    )


def upsampling_tanh_block(in_channels, out_channels, mode='nearest', scale_factor=2.0, padding=0):
    return nn.Sequential(
        nn.Upsample(mode=mode, scale_factor=scale_factor),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=padding),
        nn.Tanh()
    )


def deconv_tanh_block(in_channels, out_channels, kernel_size, stride, padding=0):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.Tanh()
    )


def one_by_one_conv_lrelu_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
        nn.LeakyReLU()
    )


class DilatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding):
        super(DilatedResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_1 = PartialConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, multi_channel=True)
        self.in_1 = nn.InstanceNorm2d(num_features=out_channels)
        self.conv_2 = PartialConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, multi_channel=True)
        self.in_2 = nn.InstanceNorm2d(num_features=out_channels)
        self.dropout = nn.Dropout2d(p=0.3)

    def forward(self, x, mask):
        residual = x

        x = self.conv_1(x, mask)
        x = self.in_1(x)
        x = F.relu(x)
        x = self.conv_2(x, mask)
        x = self.in_2(x)
        x += residual
        x = self.dropout((F.relu(x)))

        return x