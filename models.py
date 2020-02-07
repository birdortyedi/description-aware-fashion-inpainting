import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from collections import namedtuple


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
        out : self attention value + input feature attention: B X N X N (N: width*height)
        """
        b_size, C, w, h = x.size()
        p_query = self.query_conv(x).view(b_size, -1, w * h).permute(0, 2, 1) # B X C X (N)
        p_key = self.key_conv(x).view(b_size, -1, w * h) # B X C x (*W*H)
        energy = torch.bmm(p_query, p_key) # transpose check
        attention = self.softmax(energy) # B X (N) X (N)
        p_value = self.value_conv(x).view(b_size, -1, w * h) # B X C X N
        out = torch.bmm(p_value, attention.permute(0, 2, 1))
        out = out.view(b_size, C, w, h)
        out = self.gamma * out + x

        return out, attention


class DilatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding):
        super(DilatedResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, dilation=dilation)
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

class LocalDiscriminator(nn.Module):
    def __init__(self):
        super(LocalDiscriminator, self).__init__()
        self.d_block_1 = self._conv_lrelu_block(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.d_block_2 = self._conv_bn_lrelu_block(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.d_block_3 = self._conv_bn_lrelu_block(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.d_block_4 = self._conv_bn_lrelu_block(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.d_block_5 = self._conv_bn_lrelu_block(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.d_block_6 = self._conv_bn_lrelu_block(in_channels=192, out_channels=64, kernel_size=3, stride=2, padding=1)
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

    @staticmethod
    def _conv_in_lrelu_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
        return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
        stride=stride, padding=padding), nn.InstanceNorm2d(num_features=out_channels), nn.LeakyReLU(negative_slope=0.2)
        )

    @staticmethod
    def _conv_bn_lrelu_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
        return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
        stride=stride, padding=padding), nn.BatchNorm2d(num_features=out_channels), nn.LeakyReLU(negative_slope=0.2)
        )

    @staticmethod
    def _conv_lrelu_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
        return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
        stride=stride, padding=padding), nn.LeakyReLU(negative_slope=0.2)
        )


class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super(GlobalDiscriminator, self).__init__()
        # Discriminator
        self.d_block_1 = self._conv_lrelu_block(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=1)
        self.d_block_2 = self._conv_bn_lrelu_block(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.d_block_3 = self._conv_bn_lrelu_block(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=1)
        self.d_block_4 = self._conv_bn_lrelu_block(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.d_block_5 = self._conv_bn_lrelu_block(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.d_block_6 = self._conv_bn_lrelu_block(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1)
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

    @staticmethod
    def _conv_in_lrelu_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
        return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
        stride=stride, padding=padding), nn.InstanceNorm2d(num_features=out_channels), nn.LeakyReLU(negative_slope=0.2)
        )

    @staticmethod
    def _conv_bn_lrelu_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
        return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
        stride=stride, padding=padding), nn.BatchNorm2d(num_features=out_channels), nn.LeakyReLU(negative_slope=0.2)
        )

    @staticmethod
    def _conv_lrelu_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
        return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
        stride=stride, padding=padding), nn.LeakyReLU(negative_slope=0.2)
        )


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Encoder
        self.block_1 = self._conv_in_lrelu_block(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3)
        self.block_2 = self._conv_in_lrelu_block(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.block_3 = self._conv_in_lrelu_block(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        # Dilated Residual Blocks
        self.dilated_res_blocks = self._dilated_res_blocks(num_features=128, kernel_size=5, padding=4) # Self-Attention
        self.self_attention = SelfAttention(in_channels=128)
        # Visual features for concatenating with textual features
        self.block_4 = self._conv_in_lrelu_block(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.block_8 = self._deconv_in_lrelu_block(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1)
        self._1x1conv_8 = self._1x1conv_lrelu_block(in_channels=384, out_channels=64)

        self.block_9 = self._deconv_in_lrelu_block(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self._1x1conv_9 = self._1x1conv_lrelu_block(in_channels=128, out_channels=32)
        self.block_10 = self._deconv_in_lrelu_block(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.block_11 = self._deconv_tanh_block(in_channels=64, out_channels=3, kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=0.3)

    @staticmethod
    def _conv_in_lrelu_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
        return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
        stride=stride, padding=padding), nn.InstanceNorm2d(num_features=out_channels, affine=True), nn.LeakyReLU(negative_slope=0.2)
        )

    @staticmethod
    def _conv_bn_lrelu_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
        return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
        stride=stride, padding=padding), nn.BatchNorm2d(num_features=out_channels), nn.LeakyReLU(negative_slope=0.2)
        )

    @staticmethod
    def _dilated_res_blocks(num_features, kernel_size=3, stride=1, dilation=2, padding=2):
        return nn.Sequential(
        DilatedResidualBlock(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size,
        stride=stride, dilation=dilation, padding=padding),
        DilatedResidualBlock(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size,
        stride=stride, dilation=dilation, padding=padding),
        DilatedResidualBlock(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size,
        stride=stride, dilation=dilation, padding=padding),
        DilatedResidualBlock(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size,
        stride=stride, dilation=dilation, padding=padding),
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
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=padding), nn.InstanceNorm2d(num_features=out_channels, affine=True),
        nn.LeakyReLU(negative_slope=0.2)
        )

    @staticmethod
    def _deconv_in_lrelu_block(in_channels, out_channels, kernel_size, stride=2, padding=0):
        return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=0),
        nn.InstanceNorm2d(num_features=out_channels, affine=True),
        nn.LeakyReLU(negative_slope=0.2)
        )

    @staticmethod
    def _upsampling_bn_lrelu_block(in_channels, out_channels, mode='bilinear', scale_factor=2.0, padding=0):
        return nn.Sequential(
        nn.Upsample(mode=mode, scale_factor=scale_factor),
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=padding), nn.BatchNorm2d(num_features=out_channels),
        nn.LeakyReLU(negative_slope=0.2)
        )

    @staticmethod
    def _upsampling_tanh_block(in_channels, out_channels, mode='bilinear', scale_factor=2.0, padding=0):
        return nn.Sequential(
        nn.Upsample(mode=mode, scale_factor=scale_factor),
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=padding), nn.Tanh()
        )

    @staticmethod
    def _deconv_tanh_block(in_channels, out_channels, kernel_size, stride, padding=0):
        return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
        stride=stride, padding=padding), nn.Tanh()
        )

    @staticmethod
    def _1x1conv_lrelu_block(in_channels, out_channels):
        return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1), nn.LeakyReLU()
        )

class RefineNet(Net):
    def __init__(self):
        super(RefineNet, self).__init__()
        # Encoder c'ing
        self.block_5 = self._conv_in_lrelu_block(in_channels=64, out_channels=256, kernel_size=5, stride=2, padding=2)
        # Decoder
        self.block_6 = self._deconv_in_lrelu_block(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)
        self._1x1conv_6 = self._1x1conv_lrelu_block(in_channels=288, out_channels=64)
        self.block_7 = self._deconv_in_lrelu_block(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)

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


class CoarseNet(Net):
    def __init__(self, vocab_size):
        super(CoarseNet, self).__init__()
        self.block_5 = self._conv_in_lrelu_block(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        # LSTM
        self.lstm_block = self._lstm_block(vocab_size)
        # Decoder
        self.block_6 = self._deconv_in_lrelu_block(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1)
        self._1x1conv_6 = self._1x1conv_lrelu_block(in_channels=144, out_channels=32)
        self.block_7 = self._deconv_in_lrelu_block(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)
        self._1x1conv_7 = self._1x1conv_lrelu_block(in_channels=96, out_channels=128)

    def forward(self, x, descriptions):
        x_1 = self.block_1(x)
        x_2 = self.dropout(self.block_2(x_1))
        x_3 = self.dropout(self.block_3(x_2))
        dil_res_x_3 = self.dilated_res_blocks(x_3)
        attention_map, _ = self.self_attention(dil_res_x_3)
        x_4 = self.dropout(self.block_4(x_3))
        x_5 = self.dropout(self.block_5(x_4))

        visual_embedding = self.avg_pooling(x_5).squeeze()
        textual_embedding = self.lstm_block(descriptions)
        embedding = torch.cat((visual_embedding, textual_embedding), dim=1)
        x_6 = self.block_6(embedding.view(-1, 16, 4, 4))
        x_6 = torch.cat((x_5, x_6), dim=1)
        x_6 = self.dropout(self._1x1conv_6(x_6))
        x_7 = self.block_7(x_6)
        x_7 = torch.cat((x_4, x_7), dim=1)
        x_7 = self.dropout(self._1x1conv_7(x_7))
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