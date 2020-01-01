import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from utils import HDF5Dataset, RandomCentralErasing


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_11 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1)  # B x 3 x 256 x 256 --> B x 32 x 252 x 252
        # self.conv_12 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1)  # B x 64 x 252 x 252 --> B x 64 x 248 x 248
        # self.conv_13 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1)  # B x 64 x 248 x 248 --> B x 64 x 244 x 244

        self.bn_1 = nn.BatchNorm2d(num_features=32)
        self.dropout_1 = nn.Dropout2d(p=0.25)
        self.maxp_1 = nn.MaxPool2d(kernel_size=(2, 2))  # B x 32 x 252 x 252 --> B x 32 x 126 x 126

        self.conv_21 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)  # B x 32 x 126 x 126 --> B x 64 x 124 x 124
        # self.conv_22 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1)  # B x 128 x 118 x 118 --> B x 128 x 114 x 114
        # self.conv_23 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1)  # B x 128 x 114 x 114 --> B x 128 x 112 x 112

        self.bn_2 = nn.BatchNorm2d(num_features=64)
        self.dropout_2 = nn.Dropout2d(p=0.25)
        self.maxp_2 = nn.MaxPool2d(kernel_size=(2, 2))  # B x 64 x 124 x 124 --> B x 64 x 62 x 62

        self.conv_31 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)  # B x 64 x 62 x 62 --> B x 128 x 60 x 60
        # self.conv_32 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1)  # B x 256 x 52 x 52 --> B x 256 x 50 x 50
        # self.conv_33 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1)  # B x 256 x 50 x 50 --> B x 256 x 48 x 48

        self.bn_3 = nn.BatchNorm2d(num_features=128)
        self.dropout_3 = nn.Dropout2d(p=0.25)
        self.maxp_3 = nn.MaxPool2d(kernel_size=(2, 2))  # B x 128 x 60 x 60 --> B x 128 x 30 x 30

        self.conv_41 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)  # B x 128 x 30 x 30 --> B x 256 x 28 x 28
        # self.conv_42 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1)  # B x 128 x 22 x 22 --> B x 128 x 20 x 20
        # self.conv_43 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1)  # B x 128 x 20 x 20 --> B x 128 x 16 x 16

        self.bn_4 = nn.BatchNorm2d(num_features=256)
        self.dropout_4 = nn.Dropout2d(p=0.25)
        self.maxp_4 = nn.MaxPool2d(kernel_size=(2, 2))  # B x 256 x 28 x 28 --> B x 256 x 14 x 14

        self.conv_51 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1)  # B x 256 x 14 x 14 --> B x 128 x 10 x 10
        self.conv_52 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1)  # B x 128 x 10 x 10 --> B x 64 x 8 x 8

        self.bn_5 = nn.BatchNorm2d(num_features=64)
        self.dropout_5 = nn.Dropout2d(p=0.25)
        self.maxp_5 = nn.MaxPool2d(kernel_size=(2, 2))  # B x 64 x 8 x 8 --> B x 64 x 4 x 4

        self.linear_1 = nn.Linear(in_features=64 * 4 * 4, out_features=128)  # B x 64 x 4 x 4 --> B x 512

    def forward(self, x):
        x = F.leaky_relu(self.conv_11(x))
        # x = F.leaky_relu(self.conv_12(x))
        # x = F.leaky_relu(self.conv_13(x))
        x = self.bn_1(x)
        x = self.dropout_1(x)
        x = self.maxp_1(x)

        x = F.leaky_relu(self.conv_21(x))
        # x = F.leaky_relu(self.conv_22(x))
        # x = F.leaky_relu(self.conv_23(x))
        x = self.bn_2(x)
        x = self.dropout_2(x)
        x = self.maxp_2(x)

        x = F.leaky_relu(self.conv_31(x))
        # x = F.leaky_relu(self.conv_32(x))
        # x = F.leaky_relu(self.conv_33(x))
        x = self.bn_3(x)
        x = self.dropout_3(x)
        x = self.maxp_3(x)

        x = F.leaky_relu(self.conv_41(x))
        # x = F.leaky_relu(self.conv_42(x))
        # x = F.leaky_relu(self.conv_43(x))
        x = self.bn_4(x)
        x = self.dropout_4(x)
        x = self.maxp_4(x)

        x = F.leaky_relu(self.conv_51(x))
        x = F.leaky_relu(self.conv_52(x))
        x = self.bn_5(x)
        x = self.dropout_5(x)
        x = self.maxp_5(x)

        # x = F.relu(self.linear_1(torch.flatten(x, 1)))
        x = torch.flatten(x, 1)
        x = F.relu(self.linear_1(x))

        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear_1 = nn.Linear(in_features=128, out_features=512)
        self.up_1 = nn.Upsample(mode='bilinear', scale_factor=2)
        self.up_conv_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)   # B x 64 x 4 x 4 --> B x 64 x 8 x 8
        self.bn_1 = nn.BatchNorm2d(64)

        self.up_2 = nn.Upsample(mode='bilinear', scale_factor=2)
        self.up_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)  # B x 64 x 8 x 8 --> B x 128 x 16 x 16
        self.bn_2 = nn.BatchNorm2d(128)

        self.up_3 = nn.Upsample(mode='bilinear', scale_factor=2)
        self.up_conv_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)  # B x 128 x 16 x 16 --> B x 256 x 32 x 32
        self.bn_3 = nn.BatchNorm2d(256)

        self.up_4 = nn.Upsample(mode='bilinear', scale_factor=2)
        self.up_conv_4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)  # B x 256 x 32 x 32 --> B x 128 x 64 x 64
        self.bn_4 = nn.BatchNorm2d(128)

        self.up_5 = nn.Upsample(mode='bilinear', scale_factor=2)
        self.up_conv_5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)  # B x 128 x 64 x 64 --> B x 64 x 128 x 128
        self.bn_5 = nn.BatchNorm2d(64)

        self.up_6 = nn.Upsample(mode='bilinear', scale_factor=2)
        self.up_conv_6 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)   # B x 64 x 128 x 128 --> B x 3 x 256 x 256

        # TRANSPOSE CONV2D
        # self.transpose_conv_1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=5,
        #                                            stride=1, padding=0, bias=False)
        #
        # self.transpose_conv_2 = nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=4,
        #                                            stride=2, padding=1, bias=False)
        # self.bn_2 = nn.BatchNorm2d(128)
        # self.transpose_conv_3 = nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=4,
        #                                            stride=2, padding=1, bias=False)
        # self.bn_3 = nn.BatchNorm2d(256)
        # self.transpose_conv_4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4,
        #                                            stride=2, padding=1, bias=False)
        # self.bn_4 = nn.BatchNorm2d(128)
        # self.transpose_conv_5 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4,
        #                                            stride=2, padding=1, bias=False)
        # self.bn_5 = nn.BatchNorm2d(64)
        # self.transpose_conv_6 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4,
        #                                            stride=2, padding=1, bias=False)
        #

    def forward(self, x):
        x = F.leaky_relu(self.linear_1(x))
        x = F.leaky_relu(self.bn_1(self.up_conv_1(self.up_1(x.view(-1, 64, 4, 4)))))
        x = F.leaky_relu(self.bn_2(self.up_conv_2(self.up_2(x))))
        x = F.leaky_relu(self.bn_3(self.up_conv_3(self.up_3(x))))
        x = F.leaky_relu(self.bn_4(self.up_conv_4(self.up_4(x))))
        x = F.leaky_relu(self.bn_5(self.up_conv_5(self.up_5(x))))
        x = torch.sigmoid(self.up_conv_6(self.up_6(x)))

        return x


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
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(DilatedResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=dilation, dilation=dilation)
        self.in_1 = nn.InstanceNorm2d(num_features=out_channels)
        self.conv_2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        self.in_2 = nn.InstanceNorm2d(num_features=out_channels)

    def forward(self, x):
        residual = x

        x = self.conv_1(x)
        x = self.in_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = self.in_2(x)
        x += residual
        x = F.relu(x)

        return x


class LocalDiscriminator(nn.Module):
    def __init__(self):
        super(LocalDiscriminator, self).__init__()
        self.d_block_1 = self._conv_in_lrelu_block(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.d_block_2 = self._conv_in_lrelu_block(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.d_block_3 = self._conv_in_lrelu_block(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.d_block_4 = self._conv_in_lrelu_block(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.d_block_5 = self._conv_in_lrelu_block(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.d_block_6 = self._conv_in_lrelu_block(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.d_block_7 = self._conv_in_lrelu_block(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.d_block_8 = self._conv_in_lrelu_block(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.d_block_9 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        x = self.d_block_1(x)
        print(x.size())
        x = self.d_block_2(x)
        print(x.size())
        x = self.d_block_3(x)
        print(x.size())
        x = self.d_block_4(x)
        print(x.size())
        x = self.d_block_5(x)
        print(x.size())
        x = self.d_block_6(x)
        print(x.size())
        x = self.d_block_7(x)
        print(x.size())
        x = self.d_block_8(x)
        print(x.size())
        x = self.avg_pooling(x).squeeze()
        print(x.size())
        x = torch.sigmoid(self.d_block_9(x))
        return x

    @staticmethod
    def _conv_in_lrelu_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )


class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super(GlobalDiscriminator, self).__init__()
        # Discriminator
        self.d_block_1 = self._conv_in_lrelu_block(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=1)
        self.d_block_2 = self._conv_in_lrelu_block(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.d_block_3 = self._conv_in_lrelu_block(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=1)
        self.d_block_4 = self._conv_in_lrelu_block(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.d_block_5 = self._conv_in_lrelu_block(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.d_block_6 = self._conv_in_lrelu_block(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.d_block_7 = nn.Linear(in_features=16, out_features=1)

    def forward(self, x):
        x = self.d_block_1(x)
        x = self.d_block_2(x)
        x = self.d_block_3(x)
        x = self.d_block_4(x)
        x = self.d_block_5(x)
        x = self.d_block_6(x)
        x = self.avg_pooling(x)
        x = torch.sigmoid(self.d_block_7(x.squeeze()))
        return x

    @staticmethod
    def _conv_in_lrelu_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Encoder
        self.block_1 = self._conv_in_lrelu_block(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3)
        self.block_2 = self._conv_in_lrelu_block(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.block_3 = self._conv_in_lrelu_block(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)

        # Dilated Residual Blocks
        self.dilated_res_blocks = self._dilated_res_blocks(num_features=128, kernel_size=3)

        # Visual features for concatenating with textual features
        self.block_4 = self._conv_in_lrelu_block(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.block_8 = self._upsampling_in_lrelu_block(in_channels=128, out_channels=128)
        self.block_9 = self._upsampling_in_lrelu_block(in_channels=64, out_channels=64)
        self._1x1conv_9 = self._1x1conv_lrelu_block(in_channels=128, out_channels=32)
        self.block_10 = self._upsampling_in_lrelu_block(in_channels=32, out_channels=32)
        self.block_11 = self._upsampling_tanh_block(in_channels=64, out_channels=3)

    def forward(self, x, descriptions):
        raise NotImplementedError
        pass

    @staticmethod
    def _conv_in_lrelu_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    @staticmethod
    def _dilated_res_blocks(num_features, kernel_size, stride=1, dilation=2):
        return nn.Sequential(
            DilatedResidualBlock(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, stride=stride, dilation=dilation),
            DilatedResidualBlock(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, stride=stride, dilation=dilation),
            DilatedResidualBlock(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, stride=stride, dilation=dilation),
            DilatedResidualBlock(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, stride=stride, dilation=dilation),
            DilatedResidualBlock(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, stride=stride, dilation=dilation)
        )

    @staticmethod
    def _lstm_block(vocab_size, embedding_dim=32, hidden_dim=1024, n_layers=3, output_size=128):
        return nn.Sequential(
            LSTMModule(vocab_size, embedding_dim, hidden_dim, n_layers, output_size)
        )

    @staticmethod
    def _upsampling_in_lrelu_block(in_channels, out_channels, mode='bilinear', scale_factor=2.0, padding=0):
        return nn.Sequential(
            nn.Upsample(mode=mode, scale_factor=scale_factor),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=padding),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    @staticmethod
    def _upsampling_tanh_block(in_channels, out_channels, mode='bilinear', scale_factor=2.0, padding=0):
        return nn.Sequential(
            nn.Upsample(mode=mode, scale_factor=scale_factor),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=padding),
            nn.Tanh()
        )

    @staticmethod
    def _1x1conv_lrelu_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.LeakyReLU()
        )


class RefineNet(Net):
    def __init__(self):
        super(RefineNet, self).__init__()

        # Encoder c'ing
        self.block_5 = self._conv_in_lrelu_block(in_channels=64, out_channels=256, kernel_size=5, stride=2, padding=2)

        # Decoder
        self.block_6 = self._upsampling_in_lrelu_block(in_channels=16, out_channels=32)
        self._1x1conv_6 = self._1x1conv_lrelu_block(in_channels=32, out_channels=64)
        self.block_7 = self._upsampling_in_lrelu_block(in_channels=64, out_channels=64)
        self._1x1conv_7 = self._1x1conv_lrelu_block(in_channels=64, out_channels=128)

        self._1x1conv_8 = self._1x1conv_lrelu_block(in_channels=256, out_channels=64)

    def forward(self, x, descriptions):
        x_1 = self.block_1(x)
        x_2 = self.block_2(x_1)
        x_3 = self.block_3(x_2)

        dil_res_x_3 = self.dilated_res_blocks(x_3)

        x_4 = self.block_4(dil_res_x_3)
        x_5 = self.block_5(x_4)

        visual_embedding = self.avg_pooling(x_5).squeeze()

        x_6 = self.block_6(visual_embedding.view(-1, 16, 4, 4))
        x_6 = self._1x1conv_6(x_6)

        x_7 = self.block_7(x_6)
        x_7 = self._1x1conv_7(x_7)

        x_8 = self.block_8(x_7)
        x_8 = torch.cat((x_3, x_8), dim=1)
        x_8 = self._1x1conv_8(x_8)

        x_9 = self.block_9(x_8)
        x_9 = torch.cat((x_2, x_9), dim=1)
        x_9 = self._1x1conv_9(x_9)

        x_10 = self.block_10(x_9)
        x_10 = torch.cat((x_1, x_10), dim=1)

        x_11 = self.block_11(x_10)

        return x_11


class CoarseNet(Net):
    def __init__(self, vocab_size):
        super(CoarseNet, self).__init__()

        # Self-Attention
        self.self_attention = SelfAttention(in_channels=128)

        self.block_5 = self._conv_in_lrelu_block(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)

        # LSTM
        self.lstm_block = self._lstm_block(vocab_size)

        # Decoder
        self.block_6 = self._upsampling_in_lrelu_block(in_channels=16, out_channels=16)
        self._1x1conv_6 = self._1x1conv_lrelu_block(in_channels=16, out_channels=32)
        self.block_7 = self._upsampling_in_lrelu_block(in_channels=32, out_channels=32)
        self._1x1conv_7 = self._1x1conv_lrelu_block(in_channels=32, out_channels=128)

        self._1x1conv_8 = self._1x1conv_lrelu_block(in_channels=384, out_channels=64)

    def forward(self, x, descriptions):
        x_1 = self.block_1(x)
        x_2 = self.block_2(x_1)
        x_3 = self.block_3(x_2)

        dil_res_x_3 = self.dilated_res_blocks(x_3)
        attention_map, _ = self.self_attention(dil_res_x_3)

        x_4 = self.block_4(x_3)
        x_5 = self.block_5(x_4)

        visual_embedding = self.avg_pooling(x_5).squeeze()
        textual_embedding = self.lstm_block(descriptions)
        embedding = torch.cat((visual_embedding, textual_embedding), dim=1)

        x_6 = self.block_6(embedding.view(-1, 16, 4, 4))
        x_6 = self._1x1conv_6(x_6)

        x_7 = self.block_7(x_6)
        x_7 = self._1x1conv_7(x_7)

        x_8 = self.block_8(x_7)
        x_8 = torch.cat((x_3, x_8, attention_map), dim=1)
        x_8 = self._1x1conv_8(x_8)

        x_9 = self.block_9(x_8)
        x_9 = torch.cat((x_2, x_9), dim=1)
        x_9 = self._1x1conv_9(x_9)

        x_10 = self.block_10(x_9)
        x_10 = torch.cat((x_1, x_10), dim=1)

        x_11 = self.block_11(x_10)

        return x_11


class AdvancedNet(nn.Module):
    def __init__(self, vocab_size):
        super(AdvancedNet, self).__init__()
        # Encoder
        self.block_1 = self._conv_in_lrelu_block(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3)
        self.block_2 = self._conv_in_lrelu_block(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.block_3 = self._conv_in_lrelu_block(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)

        self.dilated_res_blocks = self._dilated_res_blocks(num_features=128, kernel_size=3)

        self.block_4 = self._conv_in_lrelu_block(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.block_5 = self._conv_in_lrelu_block(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.block_6 = self._conv_in_lrelu_block(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2)

        self.image_embedding_layer_1 = self._linear_block(in_features=16 * 4 * 4, out_features=256, hidden_features=128)  # 128, 256 changed to 256, 128

        # LSTM
        self.lstm_block = self._lstm_block(vocab_size)

        # Decoder
        self.block_7 = self._upsampling_in_lrelu_block(in_channels=16, out_channels=32)
        self.block_8 = self._upsampling_in_lrelu_block(in_channels=64, out_channels=64)
        self.block_9 = self._upsampling_in_lrelu_block(in_channels=128, out_channels=128)

        self.block_10 = self._upsampling_in_lrelu_block(in_channels=384, out_channels=128)
        self._1x1conv_10 = self._1x1conv_lrelu_block(in_channels=192, out_channels=128)
        self.block_11 = self._upsampling_in_lrelu_block(in_channels=128, out_channels=64)
        self._1x1conv_11 = self._1x1conv_lrelu_block(in_channels=96, out_channels=32)
        self.block_12 = self._upsampling_tanh_block(in_channels=32, out_channels=3)

    def forward(self, x, descriptions):
        x_1, x_2, x_3, x_smap, x_4, x_5, x_6 = self.extractor(x)

        x_with_descriptor = self.concat_with_descriptor(x_6, descriptions)

        x = self.block_7(x_with_descriptor.view(-1, 16, 4, 4))  # output size: torch.Size([64, 32, 8, 8])
        x = torch.cat((x, x_5), dim=1)  # output size: torch.Size([64, 64, 8, 8])
        x = self.block_8(x)  # output size: torch.Size([64, 64, 16, 16])
        x = torch.cat((x, x_4), dim=1)  # output size: torch.Size([64, 128, 16, 16])
        x = self.block_9(x)  # output size: torch.Size([64, 128, 32, 32])
        x = torch.cat((x, x_3), dim=1)

        x = torch.cat((x, x_smap), dim=1)  # output size: torch.Size([64, 384, 32, 32])

        x = self.block_10(x)  # output size: torch.Size([64, 128, 64, 64])
        x = torch.cat((x, x_2), dim=1)
        x = self._1x1conv_10(x)
        x = self.block_11(x)  # output size: torch.Size([64, 64, 128, 128])
        x = torch.cat((x, x_1), dim=1)
        x = self._1x1conv_11(x)
        x = self.block_12(x)  # output size: torch.Size([64, 3, 256, 256])

        return x

    def concat_with_descriptor(self, x_6, descriptions):
        x_ = torch.flatten(x_6, 1)  # output size: torch.Size([64, 256])
        x_ = F.relu(self.image_embedding_layer_1(x_))  # output size: torch.Size([64, 128])
        # descriptions = self.lstm_block(descriptions)
        # x_ = torch.cat((x_, descriptions), dim=1)  # output size: torch.Size([64, 256])

        return x_

    def extractor(self, x):
        x_1 = self.block_1(x)  # output size: torch.Size([64, 32, 128, 128])
        x_2 = self.block_2(x_1)  # output size: torch.Size([64, 64, 64, 64])
        x_3 = self.block_3(x_2)  # output size: torch.Size([64, 128, 32, 32])

        spatial_map, _ = self.dilated_res_blocks(x_3)  # output size: torch.Size([64, 128, 32, 32])

        x_4 = self.block_4(spatial_map)  # output size: torch.Size([64, 64, 16, 16])
        x_5 = self.block_5(x_4)  # output size: torch.Size([64, 32, 8, 8])
        x_6 = self.block_6(x_5)  # output size: torch.Size([64, 16, 4, 4])

        return x_1, x_2, x_3, spatial_map, x_4, x_5, x_6

    @staticmethod
    def _conv_in_lrelu_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.2)
            )

    @staticmethod
    def _dilated_res_blocks(num_features, kernel_size, stride=1, dilation=2):
        return nn.Sequential(
            DilatedResidualBlock(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, stride=stride, dilation=dilation),
            DilatedResidualBlock(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, stride=stride, dilation=dilation),
            DilatedResidualBlock(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, stride=stride, dilation=dilation),
            DilatedResidualBlock(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, stride=stride, dilation=dilation),
            DilatedResidualBlock(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, stride=stride, dilation=dilation)
        )

    @staticmethod
    def _lstm_block(vocab_size, embedding_dim=32, hidden_dim=1024, n_layers=3, output_size=128):
        return nn.Sequential(
            LSTMModule(vocab_size, embedding_dim, hidden_dim, n_layers, output_size)
        )

    @staticmethod
    def _linear_block(in_features, out_features, hidden_features=512):
        return nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features, out_features=out_features),
        )

    @staticmethod
    def _upsampling_in_lrelu_block(in_channels, out_channels, mode='bilinear', scale_factor=2.0, padding=0):
        return nn.Sequential(
            nn.Upsample(mode=mode, scale_factor=scale_factor),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=padding),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    @staticmethod
    def _upsampling_tanh_block(in_channels, out_channels, mode='bilinear', scale_factor=2.0, padding=0):
        return nn.Sequential(
            nn.Upsample(mode=mode, scale_factor=scale_factor),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=padding),
            nn.Tanh()
        )

    @staticmethod
    def _1x1conv_lrelu_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.LeakyReLU()
        )


if __name__ == '__main__':
    train_transform = transforms.Compose([transforms.ToTensor(),
                                          RandomCentralErasing(p=1.0, scale=(0.03, 0.12), ratio=(0.75, 1.25), value=1),
                                          # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                          ])

    fg_train = HDF5Dataset(filename='./Fashion-Gen/fashiongen_256_256_train.h5', transform=train_transform)
    print("Sample size in training: {}".format(len(fg_train)))

    coarse = CoarseNet(30000)
    refine = RefineNet()
    local = LocalDiscriminator()
    _global = GlobalDiscriminator()
