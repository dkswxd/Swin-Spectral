# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from mmcv.cnn import trunc_normal_init
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.cnn.utils.weight_init import constant_init
from mmcv.runner import _load_checkpoint
from mmcv.runner.base_module import BaseModule, ModuleList
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.utils import _pair as to_2tuple

from mmseg.ops import resize
from ...utils import get_root_logger
from ..builder import ATTENTION, BACKBONES
from torch.utils.checkpoint import checkpoint

class PatchMerging(BaseModule):
    """Merge patch feature map.

    Use Conv3d for PatchMerging
    Args:
        in_channels (int): The num of input channels.
        out_channels (int): The num of output channels.
        stride (int | tuple): the stride of the sliding length in the
            unfold layer. Defaults: 2. (Default to be equal with kernel_size).
        bias (bool, optional): Whether to add bias in linear layer or not.
            Defaults: False.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Defaults: dict(type='LN').
        init_cfg (dict, optional): The extra config for initialization.
            Defaults: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=(1, 2, 2),
                 bias=False,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.projection = build_conv_layer(
            dict(type='Conv3d'),
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=stride,
            stride=stride,)


        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        else:
            self.norm = None

    def forward(self, x, shw_shape):
        """
        x: x.shape -> [B, H*W, C]
        hw_shape: (H, W)
        """
        B, L, C = x.shape
        S, H, W = shw_shape
        assert L == S * H * W, 'input feature has wrong size'

        x = x.view(B, S, H, W, C).permute([0, 4, 1, 2, 3])  # B, C, S, H, W
        x = self.projection(x)
        Sn, Hn, Wn, Cn = x.shape[2], x.shape[3], x.shape[4], x.shape[1]
        x = x.permute([0, 2, 3, 4, 1]).view(B, Sn * Hn * Wn, Cn)  # B, S, H, W, C

        x = self.norm(x) if self.norm else x

        down_shw_shape = Sn, Hn, Wn
        return x, down_shw_shape


@ATTENTION.register_module()
class WindowMSA3D(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.0
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):

        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5
        self.init_cfg = init_cfg

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1)
                        * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Ws, Wh, Ww = self.window_size
        rel_index_coords = self.triple_step_seq((2 * Ww - 1) * (2 * Wh - 1), Ws, 2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def triple_step_seq(step1, len1, step2, len2, step3, len3):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        seq3 = torch.arange(0, step3 * len3, step3)
        return (seq1[:, None, None] + seq2[None, :, None] + seq3[None, None, :]).reshape(1, -1)


@ATTENTION.register_module()
class ShiftWindowMSA3D(BaseModule):
    """Shift Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        shift_size (tuple[int], optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(init_cfg)

        self.window_size = window_size
        self.shift_size = shift_size
        # assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA3D(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

        self.drop = build_dropout(dropout_layer)
        self.shw_shape = None

    def forward(self, query):
        shw_shape = self.shw_shape
        B, L, C = query.shape
        S, H, W = shw_shape
        assert L == S * H * W, 'input feature has wrong size'
        query = query.view(B, S, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        pad_b = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_d = (self.window_size[0] - S % self.window_size[0]) % self.window_size[0]
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b, 0, pad_d))
        S_pad, H_pad, W_pad = query.shape[1], query.shape[2], query.shape[3]

        # cyclic shift
        if self.shift_size[0] > 0 or self.shift_size[1] > 0 or self.shift_size[2] > 0:
            shifted_query = torch.roll(
                query,
                shifts=[-ss for ss in self.shift_size],
                dims=(1, 2, 3))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, S_pad, H_pad, W_pad, 1),
                                   device=query.device)  # 1 H W 1
            s_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0],
                              -self.shift_size[0]), slice(-self.shift_size[0], None))
            h_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1],
                              -self.shift_size[1]), slice(-self.shift_size[1], None))
            w_slices = (slice(0, -self.window_size[2]),
                        slice(-self.window_size[2],
                              -self.shift_size[2]), slice(-self.shift_size[2], None))
            cnt = 0
            for s in s_slices:
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, s, h, w, :] = cnt
                        cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size[0] * self.window_size[1] * self.window_size[2])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size[0] *
                                self.window_size[1] * self.window_size[2], C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0],
                                self.window_size[1], self.window_size[2], C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, S_pad, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size[0] > 0 or self.shift_size[1] > 0 or self.shift_size[2] > 0:
            x = torch.roll(
                shifted_x,
                shifts=self.shift_size,
                dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0 or pad_d > 0:
            x = x[:, :S, :H, :W, :].contiguous()

        x = x.view(B, S * H * W, C)

        x = self.drop(x)
        return x

    def window_reverse(self, windows, S, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, window_size, C)
            window_size (tuple[int]): Window size
            S (int): Spectral of image
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (S * H * W / window_size[0] / window_size[1] / window_size[2]))
        x = windows.view(B, S // window_size[0], H // window_size[1], W // window_size[2], window_size[0],
                         window_size[1], window_size[2], -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, S, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, S, H, W, C)
            window_size (int): window size
        Returns:
            windows: (num_windows*B, window_size, window_size, window_size, C)
        """
        B, S, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, S // window_size[0], window_size[0], H // window_size[1],
                   window_size[1], W // window_size[2], window_size[2], C)
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        windows = windows.view(-1, window_size[0], window_size[1], window_size[2], C)
        return windows


class SwinBlock(BaseModule):
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window size (int): The local window scale. Default: (1, 7, 7).
        window size spectral (int): The spactral window scale.
            Default: (33, 1, 1).
        shift (bool): whether to shift window or not. Default False.
        qkv_bias (int, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.2.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of nomalization.
            Default: dict(type='LN').
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=(1, 7, 7),
                 window_size_spectral=(33, 1, 1),
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 with_cp=True):

        super(SwinBlock, self).__init__()
        self.with_cp=with_cp
        self.init_cfg = init_cfg
        shift_size = [ws // 2 for ws in window_size]
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA3D(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size if shift else (0, 0, 0),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)

        self.attn_spec = ShiftWindowMSA3D(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size_spectral,
            shift_size=(0, 0, 0),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)
        self.shw_shape = None

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        self.attn.shw_shape = self.shw_shape
        self.attn_spec.shw_shape = self.shw_shape
        # if self.with_cp:
        #     x = checkpoint(self.attn, x) + checkpoint(self.attn_spec, x) + identity
        # else:
        #     x = self.attn(x) + self.attn_spec(x) + identity
        x = self.attn(x) + self.attn_spec(x) + identity

        identity = x
        x = self.norm2(x)
        # if self.with_cp:
        #     x = checkpoint(self.ffn, x, identity)
        # else:
        #     x = self.ffn(x, identity)
        x = self.ffn(x, identity)

        return x


class SwinBlockSequence(BaseModule):
    """Implements one stage in Swin Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window size (int): The local window scale. Default: (1, 7, 7).
        window size spectral (int): The spactral window scale.
            Default: (33, 1, 1).
        qkv_bias (int): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.2.
        downsample (BaseModule | None, optional): The downsample operation
            module. Default: None.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of nomalization.
            Default: dict(type='LN').
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 depth,
                 window_size=(1, 7, 7),
                 window_size_spectral=(33, 1, 1),
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 with_cp=True):
        super().__init__()

        self.with_cp = with_cp
        self.init_cfg = init_cfg

        drop_path_rate = drop_path_rate if isinstance(
            drop_path_rate,
            list) else [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = ModuleList()
        for i in range(depth):
            block = SwinBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                window_size_spectral=window_size_spectral,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                init_cfg=None,
                with_cp=with_cp)
            self.blocks.append(block)

        self.downsample = downsample

    def forward(self, x, shw_shape):
        for block in self.blocks:
            block.shw_shape = shw_shape
            if self.with_cp:
                x = checkpoint(block, x)
            else:
                x = block(x)

        if self.downsample:
            x_down, down_shw_shape = self.downsample(x, shw_shape)
            return x_down, down_shw_shape, x, shw_shape
        else:
            return x, shw_shape, x, shw_shape





# Modified from Pytorch-Image-Models
class PatchEmbed(BaseModule):
    """Image to Patch Embedding V2.

    We use a conv layer to implement PatchEmbed.
    Args:
        in_channels (int): The num of input channels. Default: 1
        embed_dims (int): The dimensions of embedding. Default: 96
        conv_type (dict, optional): The config dict for conv layers type
            selection. Default: 'Conv3d'.
        kernel_size (tuple[int]): The kernel_size of embedding conv. Default: (1, 4, 4).
        stride (tuple[int]): The slide stride of embedding conv.
            Default: None (Default to be equal with kernel_size).
        padding (int): The padding length of embedding conv. Default: 0.
        dilation (int): The dilation rate of embedding conv. Default: 1.
        pad_to_patch_size (bool, optional): Whether to pad feature map shape
            to multiple patch size. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
        use_spectral_aggregation(str): spectral aggregation strategy in 'Max',
            'Average', 'Token', 'Last'. Default: 'Token'
    """

    def __init__(self,
                 in_channels=1,
                 embed_dims=96,
                 conv_type='Conv3d',
                 kernel_size=(1, 4, 4),
                 stride=None,
                 padding=0,
                 dilation=1,
                 pad_to_patch_size=True,
                 norm_cfg=None,
                 init_cfg=None,
                 use_spectral_aggregation='Token'):
        super(PatchEmbed, self).__init__()

        self.embed_dims = embed_dims
        self.init_cfg = init_cfg

        if stride is None:
            stride = kernel_size

        self.pad_to_patch_size = pad_to_patch_size

        # The default setting of patch size is equal to kernel size.
        patch_size = kernel_size
        if isinstance(patch_size, int):
            patch_size = to_2tuple(patch_size)
        elif isinstance(patch_size, tuple):
            assert len(patch_size) == 3, \
                f'The size of patch should have length 3, ' \
                f'but got {len(patch_size)}'

        self.patch_size = patch_size

        # Use conv layer to embed
        self.projection = build_conv_layer(
            dict(type=conv_type),
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation)

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

        self.use_spectral_aggregation = use_spectral_aggregation
        if self.use_spectral_aggregation == 'Token':
            self.spectral_aggregation_token = nn.Parameter(data=torch.empty(embed_dims),requires_grad=True)
            init.trunc_normal_(self.spectral_aggregation_token, std=1)

    def forward(self, x):
        x = torch.unsqueeze(x, 1) # from (B, S, H, W) to (B, 1, S, H, W)

        S, H, W = x.shape[2], x.shape[3], x.shape[4]

        if self.pad_to_patch_size:
            # Modify H, W to multiple of patch size.
            if H % self.patch_size[0] != 0:
                x = F.pad(
                    x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
            if W % self.patch_size[1] != 0:
                x = F.pad(
                    x, (0, self.patch_size[1] - W % self.patch_size[1], 0, 0))


        x = self.projection(x)

        if self.use_spectral_aggregation == 'Token':
            _b, _c, _s, _h, _w = x.shape
            token = self.spectral_aggregation_token.view(1, -1, 1, 1, 1).repeat(_b, 1, 1, _h, _w)
            x = torch.cat((token, x), dim=2)


        self.DS, self.DH, self.DW = x.shape[2], x.shape[3], x.shape[4]

        x = x.flatten(2).transpose(1, 2) # from (B, C, S, H, W) to (B, C, S*H*W) to (B, S*H*W, C)

        if self.norm is not None:
            x = self.norm(x)

        return x



@BACKBONES.register_module()
class SwinSpectralTransformer(BaseModule):
    """Swin-Spectral Transformer backbone.

    This backbone is the implementation of `Swin-Spectral Transformer:
    Swin-Spectral Transformer for Hyperspectral image segmentation`_.

    Args:
        # pretrain_img_size (int | tuple[int]): The size of input image when
        #     pretrain. Defaults: 224. No pretained.
        in_channels (int): The num of input channels.
            Defaults: 3.
        embed_dims (int): The feature dimension. Default: 96.
        patch_size (int | tuple[int]): Patch size. Default: (1, 4, 4).
        window_size (tuple[int]): Window size. Default: (1, 7, 7).
        window_size_spectral (tuple[int]): Window size. Default: (33, 1, 1).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            Default: 4.
        depths (tuple[int]): Depths of each Swin Transformer stage.
            Default: (2, 2, 6, 2).
        num_heads (tuple[int]): Parallel attention heads of each Swin
            Transformer stage. Default: (3, 6, 12, 24).
        down_sample_stride (tuple[int]): The patch merging or patch embedding stride of
            each Swin Transformer stage. (In swin, we set kernel size equal to
            stride.) Default: (1, 2, 2).
        out_indices (tuple[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        qkv_bias (bool, optional): If True, add a learnable bias to query, key,
            value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        patch_norm (bool): If add a norm layer for patch embed and patch
            merging. Default: True.
        drop_rate (float): Dropout rate. Defaults: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Defaults: 0.1.
        # use_abs_pos_embed (bool): If True, add absolute position embedding to
        #     the patch embedding. Defaults: False.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer at
            output of backone. Defaults: dict(type='LN').
        # pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
        use_spectral_aggregation(str): spectral aggregation strategy in 'Max',
            'Average', 'Token', 'Last'. Default: 'Token'
    """

    def __init__(self,
                 # pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=96,
                 patch_size=(1, 4, 4),
                 window_size=(1, 7, 7),
                 window_size_spectral=(33, 1, 1),
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 down_sample_stride=(1, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 # use_abs_pos_embed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 # pretrained=None,
                 init_cfg=None,
                 with_cp=True,
                 use_spectral_aggregation='Token'):
        super(SwinSpectralTransformer, self).__init__()

        # if isinstance(pretrain_img_size, int):
        #     pretrain_img_size = to_2tuple(pretrain_img_size)
        # elif isinstance(pretrain_img_size, tuple):
        #     if len(pretrain_img_size) == 1:
        #         pretrain_img_size = to_2tuple(pretrain_img_size[0])
        #     assert len(pretrain_img_size) == 2, \
        #         f'The size of image should have length 1 or 2, ' \
        #         f'but got {len(pretrain_img_size)}'

        # if isinstance(pretrained, str) or pretrained is None:
        #     warnings.warn('DeprecationWarning: pretrained is a deprecated, '
        #                   'please use "init_cfg" instead')
        # else:
        #     raise TypeError('pretrained must be a str or None')

        num_layers = len(depths)
        self.out_indices = out_indices
        # self.use_abs_pos_embed = use_abs_pos_embed
        # self.pretrained = pretrained
        self.init_cfg = init_cfg
        self.use_spectral_aggregation = use_spectral_aggregation
        # assert strides[0] == patch_size, 'Use non-overlapping patch embed.'

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv3d',
            kernel_size=patch_size,
            stride=patch_size,
            pad_to_patch_size=True,
            norm_cfg=dict(type='LN') if patch_norm else None,
            init_cfg=None,
            use_spectral_aggregation=use_spectral_aggregation)

        # if self.use_abs_pos_embed:
        #     patch_row = pretrain_img_size[0] // patch_size
        #     patch_col = pretrain_img_size[1] // patch_size
        #     num_patches = patch_row * patch_col
        #     self.absolute_pos_embed = nn.Parameter(
        #         torch.zeros((1, num_patches, embed_dims)))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # stochastic depth
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule

        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=down_sample_stride,
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            else:
                downsample = None

            stage = SwinBlockSequence(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * in_channels,
                depth=depths[i],
                window_size=window_size,
                window_size_spectral=window_size_spectral,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[:depths[i]],
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                init_cfg=None,
                with_cp=with_cp)
            self.stages.append(stage)

            dpr = dpr[depths[i]:]
            if downsample:
                in_channels = downsample.out_channels

        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

    def init_weights(self):
        if True:
        # if self.pretrained is None:
            super().init_weights()
            # if self.use_abs_pos_embed:
            #     trunc_normal_init(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, Linear) or isinstance(m, nn.Conv3d) or \
                   isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    init.trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                elif isinstance(m, LayerNorm):
                    init.constant_(m.bias, 0)
                    init.constant_(m.weight, 1.0)
        # elif isinstance(self.pretrained, str):
        #     logger = get_root_logger()
        #     ckpt = _load_checkpoint(
        #         self.pretrained, logger=logger, map_location='cpu')
        #     if 'state_dict' in ckpt:
        #         state_dict = ckpt['state_dict']
        #     elif 'model' in ckpt:
        #         state_dict = ckpt['model']
        #     else:
        #         state_dict = ckpt
        #
        #     # strip prefix of state_dict
        #     if list(state_dict.keys())[0].startswith('module.'):
        #         state_dict = {k[7:]: v for k, v in state_dict.items()}
        #
        #     # reshape absolute position embedding
        #     if state_dict.get('absolute_pos_embed') is not None:
        #         absolute_pos_embed = state_dict['absolute_pos_embed']
        #         N1, L, C1 = absolute_pos_embed.size()
        #         N2, C2, H, W = self.absolute_pos_embed.size()
        #         if N1 != N2 or C1 != C2 or L != H * W:
        #             logger.warning('Error in loading absolute_pos_embed, pass')
        #         else:
        #             state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
        #                 N2, H, W, C2).permute(0, 3, 1, 2).contiguous()
        #
        #     # interpolate position bias table if needed
        #     relative_position_bias_table_keys = [
        #         k for k in state_dict.keys()
        #         if 'relative_position_bias_table' in k
        #     ]
        #     for table_key in relative_position_bias_table_keys:
        #         table_pretrained = state_dict[table_key]
        #         table_current = self.state_dict()[table_key]
        #         L1, nH1 = table_pretrained.size()
        #         L2, nH2 = table_current.size()
        #         if nH1 != nH2:
        #             logger.warning(f'Error in loading {table_key}, pass')
        #         else:
        #             if L1 != L2:
        #                 S1 = int(L1**0.5)
        #                 S2 = int(L2**0.5)
        #                 table_pretrained_resized = resize(
        #                     table_pretrained.permute(1, 0).reshape(
        #                         1, nH1, S1, S1),
        #                     size=(S2, S2),
        #                     mode='bicubic')
        #                 state_dict[table_key] = table_pretrained_resized.view(
        #                     nH2, L2).permute(1, 0).contiguous()
        #
        #     # load state_dict
        #     self.load_state_dict(state_dict, False)

    def forward(self, x):
        x = self.patch_embed(x)

        shw_shape = (self.patch_embed.DS, self.patch_embed.DH, self.patch_embed.DW)
        # if self.use_abs_pos_embed:
        #     x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, shw_shape, out, out_shw_shape = stage(x, shw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_shw_shape,
                               self.num_features[i]).permute(0, 4, 1,
                                                             2, 3)
                if self.use_spectral_aggregation == 'Max':
                    out = out.max(dim=2)[0]
                elif self.use_spectral_aggregation == 'Average':
                    out = out.mean(dim=2)
                elif self.use_spectral_aggregation == 'Token':
                    out = out[:, :, 0, :, :]
                elif self.use_spectral_aggregation == 'Last':
                    out = out[:, :, -1, :, :]
                outs.append(out.contiguous())

        return outs
