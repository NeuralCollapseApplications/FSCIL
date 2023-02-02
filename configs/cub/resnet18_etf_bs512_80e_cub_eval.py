_base_ = [
    '../_base_/models/resnet_etf.py',
    '../_base_/datasets/cub_fscil.py',
    '../_base_/schedules/cub_80e.py',
    '../_base_/default_runtime.py'
]

# CUB requires different inc settings
inc_start = 100
inc_end = 200
inc_step = 10

# model settings
model = dict(
    backbone=dict(
        _delete_=True,
        type='ResNet',
        depth=18,
        frozen_stages=1,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        norm_cfg=dict(type='BN', requires_grad=True),
    ),
    neck=dict(type='MLPFFNNeck', in_channels=512, out_channels=512),
    head=dict(
        type='ETFHead',
        num_classes=200,
        eval_classes=100,
        with_len=False,
    )
)

copy_list = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
step_list = (105, 110, 115, 120, 125, 130, 135, 140, 145, 150)
finetune_lr = 0.05
