_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/fray.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_4k.py'
]
model = dict(test_cfg=dict(mode='whole'))
runner = dict(type='IterBasedRunner', max_iters=1000)
