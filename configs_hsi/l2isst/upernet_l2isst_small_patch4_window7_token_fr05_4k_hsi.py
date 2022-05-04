_base_ = [
    '../_base_/models/upernet_l2iswinspectral.py', '../_base_/datasets/hsix.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_4k.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    fake_rate=0.5,
    # generator_file='data/HSI/25000_EMA.pth',
    # generator_file='data/HSI/186000_EMA.pth',
    # generator_file='data/HSI/140000_EMA.pth',
    # generator_file='data/HSI/102000_EMA.pth',
    # generator_file='data/HSI/s50000_EMA.pth',
    # generator_file='data/HSI/s75000_EMA.pth',
    # generator_file='data/HSI/s100000_EMA.pth',
    generator_file='data/HSI/s111000_EMA.pth',
    backbone=dict(
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(1, 7, 7),
        window_size_spectral=(9, 1, 1),
        patch_size=(4, 4, 4),
        drop_path_rate=0.3,
        patch_norm=True,
        with_cp=True,
        in_channels=1,
        use_spectral_aggregation='Token'
    ),
    decode_head=dict(
        in_channels=[96, 192, 384, 768],
        num_classes=2,
        norm_cfg=norm_cfg
    ),
    auxiliary_head=dict(
        in_channels=384,
        num_classes=2,
        norm_cfg=norm_cfg
    ))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0006,
    betas=(0.9, 0.999),
    weight_decay=0.001,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'token': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)