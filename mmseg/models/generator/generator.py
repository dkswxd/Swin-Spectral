import torch.nn as nn
import mmseg.models.generator.norms as norms
import torch
import torch.nn.functional as F
from ..builder import GENERATOR


@GENERATOR.register_module()
class OASIS_Generator(nn.Module):
    def __init__(self,
                 image_channels=32,
                 channels_G=64,
                 num_res_blocks=6,
                 semantic_nc=2,
                 z_dim=64,
                 no_3dnoise=False,
                 no_spectral_norm=False,
                 param_free_norm='syncbatch'):
        super().__init__()
        self.z_dim = z_dim
        self.semantic_nc = semantic_nc
        self.no_3dnoise = no_3dnoise
        self.num_res_blocks = num_res_blocks
        ch = channels_G
        self.channels = [16*ch, 16*ch, 16*ch, 8*ch, 4*ch, 2*ch, 1*ch]
        self.down_scale = 2**(num_res_blocks-1)
        self.conv_img = nn.Conv2d(self.channels[-1], image_channels, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
        self.body = nn.ModuleList([])
        for i in range(len(self.channels)-1):
            self.body.append(ResnetBlock_with_SPADE( self.channels[i],
                                                     self.channels[i+1],
                                                     semantic_nc,
                                                     z_dim,
                                                     no_3dnoise,
                                                     no_spectral_norm,
                                                     param_free_norm))
        if not self.no_3dnoise:
            self.fc = nn.Conv2d(semantic_nc + z_dim, 16 * ch, 3, padding=1)
        else:
            self.fc = nn.Conv2d(semantic_nc, 16 * ch, 3, padding=1)


    def forward(self, input, z=None):
        seg = input
        if not self.no_3dnoise:
            dev = seg.get_device()
            z = torch.randn(seg.size(0), self.z_dim, dtype=torch.float32, device=dev)
            z = z.view(z.size(0), self.z_dim, 1, 1)
            z = z.expand(z.size(0), self.z_dim, seg.size(2), seg.size(3))
            seg = torch.cat((z, seg), dim = 1)
        x = F.interpolate(seg, size=(z.shape[2] // self.down_scale, z.shape[3] // self.down_scale))
        x = self.fc(x)
        for i in range(self.num_res_blocks):
            x = self.body[i](x, seg)
            if i < self.num_res_blocks-1:
                x = self.up(x)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)
        return x


class ResnetBlock_with_SPADE(nn.Module):
    def __init__(self,
                 fin,
                 fout,
                 semantic_nc,
                 z_dim,
                 no_3dnoise,
                 no_spectral_norm,
                 param_free_norm):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        sp_norm = norms.get_spectral_norm(no_spectral_norm)
        self.conv_0 = sp_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = sp_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = sp_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        spade_conditional_input_dims = semantic_nc
        if not no_3dnoise:
            spade_conditional_input_dims += z_dim

        self.norm_0 = norms.SPADE(param_free_norm, fin, spade_conditional_input_dims)
        self.norm_1 = norms.SPADE(param_free_norm, fmiddle, spade_conditional_input_dims)
        if self.learned_shortcut:
            self.norm_s = norms.SPADE(param_free_norm, fin, spade_conditional_input_dims)
        self.activ = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, seg):
        return self.original_forward(x, seg)

    def original_forward(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        dx = self.conv_0(self.activ(self.norm_0(x, seg)))
        dx = self.conv_1(self.activ(self.norm_1(dx, seg)))
        out = x_s + dx
        return out
