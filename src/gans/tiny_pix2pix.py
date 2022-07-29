import torch
from torch import nn, autograd
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


class TinyP2PEncoder(nn.Module):

    def __init__(self, in_channels=3, add_noise=False):
        super().__init__()

        # tiny pix2pix generator encoder
        self.in_channels = in_channels
        self.conv_block1 = GenConvBlock(in_channels, 16, maxpool=False, dropout=False)
        self.conv_block2 = GenConvBlock(16, 32, maxpool=True, dropout=False)
        self.conv_block3 = GenConvBlock(32, 64, maxpool=True, dropout=False)
        self.conv_block4 = GenConvBlock(64, 128, maxpool=True, dropout=True)
        self.conv_block5 = GenConvBlock(128, 256, maxpool=True, dropout=True)

    def forward(self, x):
        enc1 = self.conv_block1(x)
        enc2 = self.conv_block2(enc1)
        enc3 = self.conv_block3(enc2)
        enc4 = self.conv_block4(enc3)
        enc5 = self.conv_block5(enc4)
        return enc1, enc2, enc3, enc4, enc5

    def add_noise_channel(self, x, num=1, bound_multiplier=1):
        # bound_multiplier is a scalar or a 1D tensor of length batch_size
        batch_size = x.size(0)
        filter_size = x.size(-1)
        shp = (batch_size, num, filter_size, filter_size)
        bound_multiplier = torch.tensor(bound_multiplier, device=x.device)
        noise = torch.rand(shp, device=x.device) * bound_multiplier.view(-1, 1, 1, 1)
        return torch.cat((x, noise), dim=1)


class TinyP2PDecoder(nn.Module):

    def __init__(self, out_channels):
        super().__init__()

        # tiny pix2pix generator decoder
        self.out_channels = out_channels
        self.upconv_block1 = UpConvBlock(256)
        self.upconv_block2 = UpConvBlock(128)
        self.upconv_block3 = UpConvBlock(64)
        self.upconv_block4 = UpConvBlock(32)
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=(3, 3), stride=(1, 1), padding='same')
        conv_he_init_keras(self.final_conv)

    def forward(self, enc_feat):
        enc1, enc2, enc3, enc4, enc5 = enc_feat
        dec1 = self.upconv_block1(enc5, enc4)
        dec2 = self.upconv_block2(dec1, enc3)
        dec3 = self.upconv_block3(dec2, enc2)
        dec4 = self.upconv_block4(dec3, enc1)
        out = torch.tanh(self.final_conv(dec4))
        return out


class TinyP2PGenerator(nn.Module):
    """adapted from here: https://github.com/vrkh1996/tiny-pix2pix"""

    def __init__(self, learn_residual, num_channels=3):
        super().__init__()
        self.learn_residual = learn_residual

        # tiny pix2pix generator encoder
        torch.manual_seed(2)
        self.encoder = TinyP2PEncoder(num_channels)
        self.decoder = TinyP2PDecoder(num_channels)

    def forward(self, x):
        # encode
        encdoer_features = self.encoder(x)
        # decode
        out = self.decoder(encdoer_features)

        if self.learn_residual:
            out = x + out
            out = torch.clamp(out, 0, 1)
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

    def __init__(self, in_channels=3, wgan=False, blocks_num=4):
        super().__init__()

        self.wgan = wgan
        self.blocks = blocks_num

        self.conv_block1 = DescConvBlock(in_channels, 64, batch_norm=False, stride=(2, 2))
        self.conv_block2 = DescConvBlock(64, 128, batch_norm=True, stride=(1, 1))
        self.conv_block3 = DescConvBlock(128, 256, batch_norm=True, stride=(1, 1))
        if blocks_num>4:
            self.conv_block3_5 = DescConvBlock(256, 256, batch_norm=True, stride=(1, 1))
        self.conv_block4 = DescConvBlock(256, 512, batch_norm=True, stride=(1, 1))
        if blocks_num>5:
            self.conv_block4_5 = DescConvBlock(512, 512, batch_norm=True, stride=(1, 1))

        self.conv_blocks = [*self._modules.values()]

        # dummy discriminator to see that the vm is winning
        if self.blocks == 0:
            feature_dim = 3
        else:
            # feature_dim = self.conv_block4.out_channels
            feature_dim = self.conv_blocks[-1].out_channels
        self.final_conv = nn.Conv2d(feature_dim, 1, kernel_size=(3, 3), stride=(1, 1), padding=1)
        conv_glorot_init_keras(self.final_conv)

    def forward(self, x):
        for i, conv_block in enumerate([*self._modules.values()]):
            if i == self.blocks:
                break
            x = conv_block(x)
        x = self.final_conv(x)
        if not self.wgan:
            x = torch.sigmoid(x)

        return x

    def calc_loss_and_acc(self, real_scores, fake_scores, r1_penalty=0, r1_penalty_weight=0):
        if self.wgan:
            # labels: real >= 0
            g_loss = -fake_scores.mean()
            g_acc = (fake_scores >= 0).float().mean()

            real_loss = torch.nn.ReLU(inplace=False)(1.0 - real_scores).mean()
            fake_loss = torch.nn.ReLU(inplace=False)(1.0 + fake_scores).mean()
            d_acc = 0.5 * ((real_scores >= 0).float().mean() + (fake_scores < 0).float().mean())
        else:
            # flipped labels, real=0
            g_loss = F.binary_cross_entropy(fake_scores, torch.zeros_like(fake_scores, device=fake_scores.device))
            g_acc = (fake_scores < 0.5).float().mean()

            real_loss = F.binary_cross_entropy(fake_scores, torch.ones_like(fake_scores, device=fake_scores.device))
            if real_scores is None: # some bug in first iteration
                real_scores = torch.zeros_like(fake_scores)
            fake_loss = F.binary_cross_entropy(real_scores, torch.zeros_like(real_scores, device=fake_scores.device))
            d_acc = 0.5 * ((real_scores <= 0.5).float().mean() + (fake_scores > 0.5).float().mean())
        d_loss = 0.5 * (real_loss + fake_loss)
        d_total_loss = d_loss + r1_penalty_weight * r1_penalty
        return {"real_loss": real_loss,
                "fake_loss": fake_loss,
                "d_loss": d_loss,
                "disc_r1_penalty": r1_penalty,
                "d_total_loss": d_total_loss,
                "d_acc": d_acc,
                "g_loss": g_loss,
                "g_acc": g_acc}

    @staticmethod
    def r1_penalty(real_pred, real_img):
        """R1 regularization for discriminator. The core idea is to
            penalize the gradient on real data alone: when the
            generator distribution produces the true data distribution
            and the discriminator is equal to 0 on the data manifold, the
            gradient penalty ensures that the discriminator cannot create
            a non-zero gradient orthogonal to the data manifold without
            suffering a loss in the GAN game.

            Ref:
            Eq. 9 in Which training methods for GANs do actually converge.
            """
        grad_real = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
        grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
        return grad_penalty


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
