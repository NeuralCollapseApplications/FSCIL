_base_ = [
    '../_base_/models/resnet_etf.py',
    '../_base_/datasets/mini_imagenet_fscil.py',
    '../_base_/schedules/mini_imagenet_500e.py',
    '../_base_/default_runtime.py'
]


# model settings
model = dict(
    mixup=0.,
    neck=dict(type='MLPFFNNeck', in_channels=640, out_channels=512),
    head=dict(type='ETFHead', in_channels=512, with_len=False),
)

copy_list = (1, 2, 3, 4, 5, 6, 7, 8, None, None)
step_list = (100, 110, 120, 130, 140, 150, 160, 170, None, None)
finetune_lr = 0.025
