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
from mmseg.models.backbones.swinspectral import ShiftWindowMSA3D, PatchEmbed, PatchMerging


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
        # if self.with_cp:
        #     x = checkpoint(self.attn, x) + checkpoint(self.attn_spec, x) + identity
        # else:
        #     x = self.attn(x) + self.attn_spec(x) + identity
        x = self.attn(x) + identity

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





@BACKBONES.register_module()
class SwinSpectralTransformerSingleMSA(BaseModule):
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
        super(SwinSpectralTransformerSingleMSA, self).__init__()

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
