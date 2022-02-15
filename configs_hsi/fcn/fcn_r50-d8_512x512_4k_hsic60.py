_base_ = [
    '../_base_/models/fcn_r50-d8.py', '../_base_/datasets/hsic60.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_4k.py'
]
model = dict(
    backbone=dict(in_channels=60),
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2))
