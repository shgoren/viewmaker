'''Core architecture and functionality of the viewmaker network.

Adapted from the transformer_net.py example below, using methods proposed in Johnson et al. 2016

Link:
https://github.com/pytorch/examples/blob/0c1654d6913f77f09c0505fb284d977d89c17c1a/fast_neural_style/neural_style/transformer_net.py
'''
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch_dct as dct

from viewmaker.src.gans.tiny_pix2pix import TinyP2PGenerator, TinyP2PEncoder, TinyP2PDecoder
from viewmaker.src.models.tps import TPSDecoder

ACTIVATIONS = {
    'relu': torch.nn.ReLU,
    'leaky_relu': torch.nn.LeakyReLU,
}


class Viewmaker(torch.nn.Module):
    '''Viewmaker network that stochastically maps a multichannel 2D input to an output of the same size.'''

    def __init__(self, num_channels=3, distortion_budget=0.05, activation='relu',
                 clamp=True, frequency_domain=False, downsample_to=False, num_res_blocks=5,
                 additive=1, multiplicative=0, tps=0, use_budget=True, budget_aware=False, image_dim=None):
        '''Initialize the Viewmaker network.

        Args:
            num_channels: Number of channels in the input (e.g. 1 for speech, 3 for images)
                Input will have shape [batch_size, num_channels, height, width]
            distortion_budget: Distortion budget of the viewmaker (epsilon, in the paper).
                Controls how strong the perturbations can be.
            activation: The activation function used in the network ('relu' and 'leaky_relu' currently supported)
            clamp: Whether to clamp the outputs to [0, 1] (useful to ensure output is, e.g., a valid image)
            frequency_domain: Whether to apply perturbation (and distortion budget) in the frequency domain.
                This is useful for shifting the inductive bias of the viewmaker towards more global / textural views.
            downsample_to: Downsamples the image, applies viewmaker, then upsamples. Possibly useful for 
                higher-resolution inputs, but not evaluaed in the paper.
            num_res_blocks: Number of residual blocks to use in the network.
        '''
        super().__init__()

        self.use_budget = use_budget
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.activation = activation
        self.clamp = clamp
        self.frequency_domain = frequency_domain
        self.downsample_to = downsample_to
        self.distortion_budget = distortion_budget
        self.act = ACTIVATIONS[activation]
        self.multiplicative = multiplicative
        self.additive = additive
        self.tps = tps
        self.budget_aware = budget_aware

        self.feature_extraction = nn.Sequential(
            ConvLayer(self.num_channels + 1, 32, kernel_size=9, stride=1),
            # ConvLayer(self.num_channels, 32, kernel_size=9, stride=1),
            torch.nn.InstanceNorm2d(32, affine=True),
            self.act(),
            ConvLayer(32, 64, kernel_size=3, stride=2),
            torch.nn.InstanceNorm2d(64, affine=True),
            self.act(),
            ConvLayer(64, 128, kernel_size=3, stride=2),
            torch.nn.InstanceNorm2d(128, affine=True),
            self.act()
        )

        self.res1 = ResidualBlock(128 + 1)
        self.res2 = ResidualBlock(128 + 2)
        self.res3 = ResidualBlock(128 + 3)
        self.res4 = ResidualBlock(128 + 4)
        self.res5 = ResidualBlock(128 + 5)
        self.res6 = ResidualBlock(128 + 6)
        self.res7 = ResidualBlock(128 + 7)
        self.res8 = ResidualBlock(128 + 8)

        last_feature_dim = (128 + self.num_res_blocks, image_dim[0] // 4, image_dim[0] // 4)
        # Upsampling Layers
        ## shahaf change ##
        # viewmaker has multiple decoders to add variability
        assert additive + multiplicative + tps > 0, "no augmentation specified"
        if additive + multiplicative >0:
            self.pixel_transforms = nn.ModuleList([self._full_size_output_net() for _ in range(self.additive + self.multiplicative)])
        else:
            self.pixel_transforms = None

        if tps > 0:
            assert image_dim is not None, "image dimensions are required for tps augmentations"

            self.geometric_transforms = nn.ModuleList([TPSDecoder(last_feature_dim, image_dim, grid_hw=(2,2)) for _ in range(self.tps)])
        else:
            self.geometric_transforms = None

        # self.dynamic_budget = False
        # self.budget_net = nn.Sequential(
        #     ConvLayer(last_feature_dim[0], 32, kernel_size=3, stride=1),
        #     torch.nn.BatchNorm2d(32),
        #     self.act(),
        #     ConvLayer(32, 16, kernel_size=3, stride=1),
        #     torch.nn.BatchNorm2d(16),
        #     self.act(),
        #     ConvLayer(16, 1, kernel_size=3, stride=1),
        # )

    def _full_size_output_net(self):
        return nn.Sequential(
            UpsampleConvLayer(128 + self.num_res_blocks + int(self.budget_aware), 64, kernel_size=3, stride=1, upsample=2),
            torch.nn.InstanceNorm2d(64, affine=True),
            self.act(),
            ResidualBlock(64),
            UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2),
            torch.nn.InstanceNorm2d(32, affine=True),
            self.act(),
            ResidualBlock(32),
            ConvLayer(32, self.num_channels, kernel_size=9,
                      stride=1)
        )

    @staticmethod
    def zero_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            # actual 0 has symmetry problems
            F.normal_(m.weight.data, mean=0, std=1e-4)
            # init.constant_(m.weight.data, 0)
            F.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            pass

    def add_noise_channel(self, x, num=1, bound_multiplier=1):
        # bound_multiplier is a scalar or a 1D tensor of length batch_size
        batch_size = x.size(0)
        filter_size = x.size(-1)
        shp = (batch_size, num, filter_size, filter_size)
        bound_multiplier = torch.tensor(bound_multiplier, device=x.device)
        noise = torch.rand(shp, device=x.device) * bound_multiplier.view(-1, 1, 1, 1)
        return torch.cat((x, noise), dim=1)

    def basic_net(self, x, num_res_blocks=5, bound_multiplier=1):
        if num_res_blocks not in list(range(9)):
            raise ValueError(f'num_res_blocks must be in {list(range(9))}, got {num_res_blocks}.')

        x_noise = self.add_noise_channel(x, bound_multiplier=bound_multiplier)
        f = self.feature_extraction(x_noise)
        # f = self.feature_extraction(x)

        # Features that could be useful for other auxilary layers / losses.
        # [batch_size, 128]
        features = f.clone().mean([-1, -2])

        for i, res in enumerate([self.res1, self.res2, self.res3, self.res4, self.res5, self.res6, self.res7, self.res8]):
            if i < num_res_blocks:
                f = res(self.add_noise_channel(f, bound_multiplier=bound_multiplier))

        # if self.dynamic_budget:
        #     self.distortion_budget = F.sigmoid(self.budget_net(f).mean())

        ## shahaf ##
        # modifiers are a generalizations of the residuals
        if self.pixel_transforms:
            if self.budget_aware:
                f = torch.cat([f, torch.full_like(f[:,[0],:,:], self.distortion_budget)], dim=1)
            modifiers = torch.stack([up(f) for up in self.pixel_transforms], dim=1)
        else:
            modifiers = None
        if self.geometric_transforms:
            geometric_views = torch.stack([up(f, x, self.distortion_budget) for up in self.geometric_transforms], dim=1)
        else:
            geometric_views = None

        return modifiers, geometric_views, features

    def get_delta(self, y_pixels, eps=1e-6):
        """
        Constrains the input perturbation by projecting it onto an L1 sphere
        :param y_pixels: (b,v,h,w)
        :param eps:
        :return:
        """

        distortion_budget = self.distortion_budget
        delta = torch.tanh(y_pixels)  # Project to [-1, 1]
        if self.use_budget:
            avg_magnitude = delta.abs().mean([1, 2, 3], keepdim=True)
            max_magnitude = distortion_budget
            delta = delta * max_magnitude / (avg_magnitude + eps)
        return delta

    def forward(self, x):
        x_orig = x
        if self.downsample_to:
            # Downsample.
            x = torch.nn.functional.interpolate(
                x, size=(self.downsample_to, self.downsample_to), mode='bilinear')
        y = x

        if self.frequency_domain:
            # Input to viewmaker is in frequency domain, outputs frequency domain perturbation.
            # Uses the Discrete Cosine Transform.
            # shape still [batch_size, C, W, H]
            y = dct.dct_2d(y)

        modifiers, geometric_views, features = self.basic_net(y, self.num_res_blocks, bound_multiplier=1)
        result = []
        # shahaf #
        # masks are multiplicative and residual are additive
        if modifiers is not None:
            masks, residuals = modifiers[:, :self.multiplicative], modifiers[:, self.multiplicative:]

            for i in range(residuals.size(1)):
                view = residuals[:, i]
                result.append(self.add_residual(x_orig, view))

            for i in range(masks.size(1)):
                view = masks[:, i]
                result.append(self.apply_learned_mask(x_orig, view))
        if geometric_views is not None:
            result.extend(geometric_views.unbind(1))

        result = torch.stack(result).transpose(0, 1).reshape(-1, *result[0].shape[1:])
        return result

    def get_mask_delta(self, y_pixels, eps=1e-4):
        """
        Constrains the input perturbation by projecting it onto an L1 sphere
        :param y_pixels: (b,v,h,w)
        :param eps:
        :return:
        """

        distortion_budget = self.distortion_budget
        delta = torch.sigmoid(y_pixels) * 2  # Project to [0, 2]
        if self.use_budget:
            # avg_magnitude = delta.mean([1, 2, 3], keepdim=True)
            avg_magnitude = (1-delta).abs().mean([1, 2, 3], keepdim=True)
            max_magnitude = distortion_budget
            # delta = 1-( max_magnitude + delta - avg_magnitude )
            delta = 1 - ((1-delta) * (max_magnitude / (avg_magnitude + eps)))
            # delta = 1-((1-delta)*max_magnitude/avg_magnitude)
        return delta
# Im = 1-((1-image)*budget/average(1-image))

    def apply_learned_mask(self, x, mask):
        delta = self.get_mask_delta(mask)
        # delta_min = delta.view(b,h*w).min(1,keepdim=True)[0].view(b,1,1,1)
        # delta_max = delta.view(b,h*w).max(1,keepdim=True)[0].view(b,1,1,1)

        # mask_tanh = torch.tanh(mask)
        # intensity_mask = torch.softmax(delta.mean(1,keepdim=True).view(b,1,h*w),-1).view(b,1,h,w)
        # # intensity_mask = torch.tanh(mask.mean(1,keepdim=True))+1
        # intensity_mask = intensity_mask / intensity_mask.max() # over all images instead of per image
        # M = intensity_mask.max()
        # m = intensity_mask.min()
        # eps = 1e-8
        # # multiplicative = (((intensity_mask+eps)/(intensity_mask+m+eps))-0.5)*(2*(M+m+eps)/(M-m+eps))
        # multiplicative = (delta - delta_min + eps) / (delta_max-delta_min + eps) + 0.6
        res = x * delta
        res = torch.clamp(res, 0, 1.0)
        return res

    def add_residual(self, x, residual):
        delta = self.get_delta(residual)
        if self.frequency_domain:
            # Compute inverse DCT from frequency domain to time domain.
            delta = dct.idct_2d(delta)
        if self.downsample_to:
            # Upsample.
            delta = torch.nn.functional.interpolate(delta, size=x.shape[-2:], mode='bilinear')

        # Additive perturbation
        result = x + delta
        if self.clamp:
            result = torch.clamp(result, 0, 1.0)
        return result



class ViewmakerPix2Pix(Viewmaker):
    '''Viewmaker network that stochastically maps a multichannel 2D input to an output of the same size.'''

    def __init__(self, num_channels=3, distortion_budget=0.05, activation='relu',
                 clamp=True, frequency_domain=False, downsample_to=False, num_res_blocks=5,
                 num_views=1, masks=0, use_budget=True):
        '''Initialize the Viewmaker network.

        Args:
            num_channels: Number of channels in the input (e.g. 1 for speech, 3 for images)
                Input will have shape [batch_size, num_channels, height, width]
            distortion_budget: Distortion budget of the viewmaker (epsilon, in the paper).
                Controls how strong the perturbations can be.
            activation: The activation function used in the network ('relu' and 'leaky_relu' currently supported)
            clamp: Whether to clamp the outputs to [0, 1] (useful to ensure output is, e.g., a valid image)
            frequency_domain: Whether to apply perturbation (and distortion budget) in the frequency domain.
                This is useful for shifting the inductive bias of the viewmaker towards more global / textural views.
            downsample_to: Downsamples the image, applies viewmaker, then upsamples. Possibly useful for
                higher-resolution inputs, but not evaluaed in the paper.
            num_res_blocks: Number of residual blocks to use in the network.
        '''
        super().__init__()

        self.use_budget = use_budget
        self.num_channels = num_channels
        self.clamp = clamp
        self.frequency_domain = frequency_domain
        self.distortion_budget = distortion_budget
        self.num_views = num_views

        torch.manual_seed(2)
        self.encoder = TinyP2PEncoder(num_channels+1)
        self.upsampling = nn.ModuleList(
            [TinyP2PDecoder(num_channels) for _ in range(self.num_views)])  # TODO: check adding instance norm

    def basic_net(self, y, num_res_blocks=5, bound_multiplier=1):
        if num_res_blocks not in list(range(6)):
            raise ValueError(f'num_res_blocks must be in {list(range(6))}, got {num_res_blocks}.')

        y = self.add_noise_channel(y, bound_multiplier=bound_multiplier)
        f = self.encoder(y)

        modifiers = torch.stack([up(f) for up in self.upsampling], dim=1)

        return modifiers, f

    def get_delta(self, delta, eps=1e-4):
        """
        Constrains the input perturbation by projecting it onto an L1 sphere
        :param y_pixels: (b,v,h,w)
        :param eps:
        :return:
        """

        distortion_budget = self.distortion_budget
        if self.use_budget:
            avg_magnitude = delta.abs().mean([1, 2, 3], keepdim=True)
            max_magnitude = distortion_budget
            delta = delta * max_magnitude / (avg_magnitude + eps)
        return delta
# ---

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels, activation='relu'):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.act = ACTIVATIONS[activation]()

    def forward(self, x):
        residual = x
        out = self.act(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(
                x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


VIEWMAKERS = {
    'ViewmakerPix2Pix': ViewmakerPix2Pix,
    'Viewmaker': Viewmaker
}
