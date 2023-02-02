# optimizer
optimizer = dict(
    type='SGD', lr=0.25, momentum=0.9, weight_decay=0.0005
)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealingCooldown',
    min_lr=None,
    min_lr_ratio=1.e-2,
    cool_down_ratio=0.1,
    cool_down_time=10,
    by_epoch=False,
    # warmup
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.1,
    warmup_by_epoch=False
)

runner = dict(type='EpochBasedRunner', max_epochs=50)
