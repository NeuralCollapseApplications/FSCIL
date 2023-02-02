_base_ = [
    '../_base_/models/resnet_etf.py',
    '../_base_/datasets/cifar_fscil.py',
    '../_base_/schedules/cifar_200e.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    mixup=0.5,
    mixup_prob=0.75,
    neck=dict(type='MLPFFNNeck', in_channels=640, out_channels=512),
    head=dict(type='ETFHead', in_channels=512, with_len=True),
)

copy_list = (1, 1, 1, 1, 1, 1, 1, 1, None, None)
step_list = (50, 75, 100, 120, 140, 160, 200, 200, None, None)
finetune_lr = 0.25
