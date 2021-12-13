import torch
from torch import nn
from torch.nn import functional as F

def conv_he_init_keras(conv):
    torch.nn.init.kaiming_normal_(conv.weight)
    torch.nn.init.zeros_(conv.bias)

def conv_glorot_init_keras(conv):
    torch.nn.init.xavier_uniform_(conv.weight)
    torch.nn.init.zeros_(conv.bias)

class GenConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, maxpool, dropout):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        else:
            self.maxpool = nn.Identity()
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(3, 3), stride=(1, 1), padding='same')
        conv_he_init_keras(self.conv1)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=(3, 3), stride=(1, 1), padding='same')
        conv_he_init_keras(self.conv2)
        if dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        x = F.relu(self.maxpool(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.dropout(x))

        return x

class UpConvBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels // 2

        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv_up = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(2, 2), stride=(1, 1), padding='same')
        conv_he_init_keras(self.conv_up)
        ##### concat ######
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(3, 3), stride=(1, 1), padding='same')
        conv_he_init_keras(self.conv1)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=(3, 3), stride=(1, 1), padding='same')
        conv_he_init_keras(self.conv2)

    def forward(self, upstream, downstream):
        x = F.relu(self.conv_up(self.upscale(upstream)))
        x = torch.cat([x, downstream], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        return x


class TinyP2PGenerator(nn.Module):
    """adapted from here: https://github.com/vrkh1996/tiny-pix2pix"""

    def __init__(self, learn_residual, in_channels=3):
        super().__init__()
        self.learn_residual = learn_residual

        # tiny pix2pix generator encoder
        self.conv_block1 = GenConvBlock(in_channels, 16, maxpool=False, dropout=False)
        self.conv_block2 = GenConvBlock(16, 32, maxpool=True, dropout=False)
        self.conv_block3 = GenConvBlock(32, 64, maxpool=True, dropout=False)
        self.conv_block4 = GenConvBlock(64, 128, maxpool=True, dropout=True)
        self.conv_block5 = GenConvBlock(128, 256, maxpool=True, dropout=True)

        # tiny pix2pix generator decoder
        self.upconv_block1 = UpConvBlock(256)
        self.upconv_block2 = UpConvBlock(128)
        self.upconv_block3 = UpConvBlock(64)
        self.upconv_block4 = UpConvBlock(32)

        self.final_conv = nn.Conv2d(16, 3, kernel_size=(3, 3), stride=(1, 1), padding='same')
        conv_he_init_keras(self.final_conv)

    def forward(self, x):
        # encode
        enc1 = self.conv_block1(x)
        enc2 = self.conv_block2(enc1)
        enc3 = self.conv_block3(enc2)
        enc4 = self.conv_block4(enc3)
        enc5 = self.conv_block5(enc4)

        # decode
        dec1 = self.upconv_block1(enc5, enc4)
        dec2 = self.upconv_block2(dec1, enc3)
        dec3 = self.upconv_block3(dec2, enc2)
        dec4 = self.upconv_block4(dec3, enc1)
        out = torch.tanh(self.final_conv(dec4))
        if self.learn_residual:
            out = x + out

        return out


class DescConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(3, 3), stride=stride, padding=1)
        conv_glorot_init_keras(self.conv1)
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(self.out_channels, momentum=0.8)
        else:
            self.batch_norm = nn.Identity()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(self.batch_norm(x), 0.2)
        x = self.dropout(x)

        return x


class TinyP2PDiscriminator(nn.Module):
    """adapted from here: https://github.com/vrkh1996/tiny-pix2pix"""

    def __init__(self, in_channels=3):
        super().__init__()

        self.conv_block1 = DescConvBlock(in_channels, 64, batch_norm=False, stride=(2, 2))
        self.conv_block2 = DescConvBlock(64, 128, batch_norm=True, stride=(1, 1))
        self.conv_block3 = DescConvBlock(128, 256, batch_norm=True, stride=(1, 1))
        self.conv_block4 = DescConvBlock(256, 512, batch_norm=True, stride=(1, 1))
        self.final_conv = nn.Conv2d(512, 1, kernel_size=(3, 3), stride=(1, 1), padding=1)
        conv_glorot_init_keras(self.final_conv)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = torch.sigmoid(self.final_conv(x))

        return x


def normalize_generated(imgs):
    """

    :param imgs: to the range of [-1,1]
    :return:
    """
    # These numbers were computed using compute_image_dset_stats.py
    mean = torch.tensor([0.491, 0.482, 0.446], device=imgs.device)
    std = torch.tensor([0.247, 0.243, 0.261], device=imgs.device)
    imgs = (imgs - mean[None, :, None, None]) / std[None, :, None, None]
    return imgs



