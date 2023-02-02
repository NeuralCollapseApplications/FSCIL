img_size = 32
_img_resize_size = 36
img_norm_cfg = dict(mean=[129.304, 124.070, 112.434], std=[68.170, 65.392, 70.418], to_rgb=False)
meta_keys = ('filename', 'ori_filename', 'ori_shape',
             'img_shape', 'flip', 'flip_direction',
             'img_norm_cfg', 'cls_id', 'img_id')

train_pipeline = [
    dict(type='RandomResizedCrop', size=img_size, scale=(0.6, 1.), interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'], meta_keys=meta_keys)
]

test_pipeline = [
    dict(type='Resize', size=(_img_resize_size, -1), interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=img_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'gt_label'], meta_keys=meta_keys)
]

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
    train_dataloader=dict(
        persistent_workers=True,
    ),
    val_dataloader=dict(
        persistent_workers=True,
    ),
    test_dataloader=dict(
        persistent_workers=True,
    ),
    train=dict(
        type='RepeatDataset',
        times=4,
        dataset=dict(
            type='CIFAR100FSCILDataset',
            data_prefix='/opt/data/cifar',
            pipeline=train_pipeline,
            num_cls=60,
            subset='train',
        )
    ),
    val=dict(
        type='CIFAR100FSCILDataset',
        data_prefix='/opt/data/cifar',
        pipeline=test_pipeline,
        num_cls=60,
        subset='test',
    ),
    test=dict(
        type='CIFAR100FSCILDataset',
        data_prefix='/opt/data/cifar',
        pipeline=test_pipeline,
        num_cls=100,
        subset='test',
    )
)
