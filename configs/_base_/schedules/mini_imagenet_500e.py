# optimizer
optimizer = dict(
    type='SGD', lr=0.25, momentum=0.9, weight_decay=0.0005
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    gamma=0.25,
    warmup='linear',
    warmup_iters=3000,
    warmup_ratio=0.25,
    step=[20, 30, 35, 40, 45]
)
runner = dict(type='EpochBasedRunner', max_epochs=50)
