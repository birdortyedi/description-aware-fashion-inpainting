import torch
import torch.nn.functional as F
from torch import nn


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
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=padding),
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


class LSTMModule(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, output_size):
        super(LSTMModule, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=0.25, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.25)
        self.linear_1 = nn.Linear(hidden_dim * 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers * 2, x.size(0), self.hidden_dim).requires_grad_().cuda()
        c0 = torch.zeros(self.n_layers * 2, x.size(0), self.hidden_dim).requires_grad_().cuda()

        x = self.embedding(x)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.dropout(lstm_out[:, -1, :])
        out = F.relu(self.linear_1(out))
        return out


class SelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels, gamma=1):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels

        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(gamma), requires_grad=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N: width*height)
        """
        b_size, C, w, h = x.size()
        p_query = self.query_conv(x).view(b_size, -1, w * h).permute(0, 2, 1)  # B X C X (N)
        p_key = self.key_conv(x).view(b_size, -1, w * h)  # B X C x (*W*H)
        energy = torch.bmm(p_query, p_key)  # transpose check
        attention = self.softmax(energy)  # B X (N) X (N)
        p_value = self.value_conv(x).view(b_size, -1, w * h)  # B X C X N

        out = torch.bmm(p_value, attention.permute(0, 2, 1))
        out = out.view(b_size, C, w, h)

        out = self.gamma * out + x
        return out, attention


class DilatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding):
        super(DilatedResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.in_1 = nn.InstanceNorm2d(num_features=out_channels)
        self.conv_2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        self.in_2 = nn.InstanceNorm2d(num_features=out_channels)
        self.dropout = nn.Dropout2d(p=0.3)

    def forward(self, x):
        residual = x

        x = self.conv_1(x)
        x = self.in_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = self.in_2(x)
        x += residual
        x = self.dropout((F.relu(x)))

        return x