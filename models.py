import torch
from torch import nn
from torch.nn import functional as F


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
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=0.25, batch_first=True)
        self.dropout = nn.Dropout(0.25)
        self.linear_1 = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).requires_grad_().cuda()
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).requires_grad_().cuda()

        x = self.embedding(x)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.dropout(lstm_out[:, -1, :])
        out = F.relu(self.linear_1(out))
        return out


class Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, lstm_hidden_dim=1024, lstm_n_layers=3, lstm_output_size=512):
        super(Net, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.lstm = LSTMModule(vocab_size, embedding_dim, lstm_hidden_dim, lstm_n_layers, lstm_output_size)

    def forward(self, x, desc):
        x = self.encoder(x)
        desc = self.lstm(desc)

        out = self.decoder(torch.cat((x, desc), dim=1))

        return out
