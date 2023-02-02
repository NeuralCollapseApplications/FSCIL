_base_ = [
    '../_base_/models/resnet_etf.py',
    '../_base_/datasets/mini_imagenet_fscil.py',
    '../_base_/schedules/mini_imagenet_500e.py',
    '../_base_/default_runtime.py'
]


# model settings
model = dict(
    neck=dict(type='MLPFFNNeck', in_channels=640, out_channels=512),
    head=dict(type='ETFHead', in_channels=512, with_len=False),
)

