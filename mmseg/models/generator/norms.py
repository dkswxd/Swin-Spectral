import torch.nn.utils.spectral_norm as spectral_norm
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer
import torch.nn as nn
import torch.nn.functional as F


class SPADE(nn.Module):
    def __init__(self, param_free_norm, norm_nc, label_nc):
        super().__init__()
        self.first_norm = get_norm_layer(param_free_norm, norm_nc)
        ks = 3
        nhidden = 128
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):
        normalized = self.first_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out


def get_spectral_norm(no_spectral_norm):
    if no_spectral_norm:
        return torch.nn.Identity()
    else:
        return spectral_norm


def get_norm_layer(param_free_norm, norm_nc):
    if param_free_norm == 'instance':
        return build_norm_layer(dict(type="IN2D", affine=False),norm_nc)[1]
    if param_free_norm == 'syncbatch':
        return build_norm_layer(dict(type="SyncBN", affine=False),norm_nc)[1]
    if param_free_norm == 'batch':
        return build_norm_layer(dict(type="BN2D", affine=False),norm_nc)[1]
    else:
        raise ValueError('%s is not a recognized param-free norm type in SPADE'
                         % param_free_norm)