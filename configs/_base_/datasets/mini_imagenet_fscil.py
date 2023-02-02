img_size = 84
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
meta_keys = ('filename', 'ori_filename', 'ori_shape',
             'img_shape', 'flip', 'flip_direction',
             'img_norm_cfg', 'cls_id', 'img_id')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=img_size),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'], meta_keys=meta_keys)
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(int(img_size * 1.15), -1)),
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
        times=10,
        dataset=dict(
            type='MiniImageNetFSCILDataset',
            data_prefix='/opt/data/miniimagenet',
            pipeline=train_pipeline,
            num_cls=60,
            subset='train',
        )
    ),
    val=dict(
        type='MiniImageNetFSCILDataset',
        data_prefix='/opt/data/miniimagenet',
        pipeline=test_pipeline,
        num_cls=60,
        subset='test',
    ),
    test=dict(
        type='MiniImageNetFSCILDataset',
        data_prefix='/opt/data/miniimagenet',
        pipeline=test_pipeline,
        num_cls=100,
        subset='test',
    )
)
