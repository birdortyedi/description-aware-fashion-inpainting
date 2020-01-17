from torchvision import transforms
from torchvision import models

from collections import namedtuple
from utils import HDF5Dataset, CentralErasing
from blocks import *


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
        self.d_block_1 = conv_lrelu_block(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=1)
        self.d_block_2 = conv_bn_lrelu_block(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.d_block_3 = conv_bn_lrelu_block(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=1)
        self.d_block_4 = conv_bn_lrelu_block(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.d_block_5 = conv_bn_lrelu_block(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.d_block_6 = conv_bn_lrelu_block(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1)
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Encoder
        self.block_1 = PartialConv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3,
                                     bias=False, return_mask=True, multi_channel=True)
        self.in_1 = nn.InstanceNorm2d(num_features=32)

        self.block_2 = PartialConv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2,
                                     bias=False, return_mask=True, multi_channel=True)
        self.in_2 = nn.InstanceNorm2d(num_features=64)

        self.block_3 = PartialConv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2,
                                     bias=False, return_mask=True, multi_channel=True)
        self.in_3 = nn.InstanceNorm2d(num_features=128)

        # Dilated Residual Blocks
        self.dilated_res_blocks = dilated_res_blocks(num_features=128, kernel_size=5, padding=4)
        # Self-Attention
        self.self_attention = SelfAttention(in_channels=128)

        # Visual features for concatenating with textual features
        self.block_4 = PartialConv2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2,
                                     bias=False, return_mask=True, multi_channel=True)
        self.in_4 = nn.InstanceNorm2d(num_features=64)

        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.block_8 = upsampling_in_lrelu_block(in_channels=128, out_channels=128, padding=1)
        self._1x1conv_8 = one_by_one_conv_lrelu_block(in_channels=384, out_channels=64)
        self.block_9 = upsampling_in_lrelu_block(in_channels=64, out_channels=64, padding=1)
        self._1x1conv_9 = one_by_one_conv_lrelu_block(in_channels=128, out_channels=32)
        self.block_10 = upsampling_in_lrelu_block(in_channels=32, out_channels=32, padding=1)
        self.block_11 = upsampling_tanh_block(in_channels=64, out_channels=3)

        self.dropout = nn.Dropout2d(p=0.3)


class RefineNet(Net):
    def __init__(self):
        super(RefineNet, self).__init__()

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


class CoarseNet(Net):
    def __init__(self, vocab_size):
        super(CoarseNet, self).__init__()

        self.block_5 = PartialConv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2,
                                     bias=False, return_mask=True, multi_channel=True)
        self.in_5 = nn.InstanceNorm2d(num_features=128)

        # LSTM
        self.lstm_block = lstm_block(vocab_size)

        # Decoder
        self.block_6 = upsampling_in_lrelu_block(in_channels=16, out_channels=16)
        self._1x1conv_6 = one_by_one_conv_lrelu_block(in_channels=144, out_channels=32)
        self.block_7 = upsampling_in_lrelu_block(in_channels=32, out_channels=32)
        self._1x1conv_7 = one_by_one_conv_lrelu_block(in_channels=96, out_channels=128)

    def forward(self, x, descriptions, mask):
        x_1, m_1 = self.block_1(x, mask)
        x_1 = F.leaky_relu(self.in_1(x_1))
        print(x_1.size())
        x_2, m_2 = self.block_2(x_1, m_1)
        x_2 = self.dropout(F.leaky_relu(self.in_2(x_2)))
        print(x_2.size())
        x_3, m_3 = self.block_3(x_2, m_2)
        x_3 = self.dropout(F.leaky_relu(self.in_3(x_3)))
        print(x_3.size())

        dil_res_x_3 = self.dilated_res_blocks(x_3)
        attention_map, _ = self.self_attention(dil_res_x_3)

        x_4, m_4 = self.block_4(x_3, m_3)
        x_4 = self.dropout(F.leaky_relu(self.in_4(x_4)))
        print(x_4.size())
        x_5, m_5 = self.block_5(x_4, m_4)
        x_5 = self.dropout(F.leaky_relu(self.in_5(x_5)))
        print(x_5.size())

        visual_embedding = self.avg_pooling(x_5).squeeze()
        print(visual_embedding.size())
        textual_embedding = self.lstm_block(descriptions)
        print(textual_embedding.size())
        embedding = torch.cat((visual_embedding, textual_embedding), dim=1)
        print(embedding.size())

        x_6 = self.block_6(embedding.view(-1, 16, 4, 4))
        print(x_6.size())
        x_6 = torch.cat((x_5, x_6), dim=1)
        print(x_6.size())
        x_6 = self.dropout(self._1x1conv_6(x_6))
        print(x_6.size())

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
