# dataset settings
dataset_type = 'HSIDataset'
data_root = 'data/HSI'
img_norm_cfg = dict(
    mean=[128]*32, std=[16]*32, to_rgb=False)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadENVIHyperSpectralImageFromFile',channel_select=range(4,36),median_blur=False),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(320, 256), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadENVIHyperSpectralImageFromFile',channel_select=range(4,36)),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(320, 256),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            # dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='hdrx_dir',
        ann_dir='annraw_dir',
        split='splitx_dir/split_{}_train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='hdrx_dir',
        ann_dir='annx_dir',
        split='splitx_dir/split_{}_test.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='hdrx_dir',
        # ann_dir='annx_dir',
        ann_dir='annraw_dir',
        split='splitx_dir/split_{}_test.txt',
        pipeline=test_pipeline))
