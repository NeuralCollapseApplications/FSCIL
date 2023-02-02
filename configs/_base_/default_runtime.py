# checkpoint saving
checkpoint_config = dict(interval=1, max_keep_ckpts=2)
evaluation = dict(interval=1, save_best='auto')
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ]
)

dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]

load_from = None
resume_from = None

# Test configs
mean_neck_feat = True
mean_cur_feat = False
feat_test = False
grad_clip = None
finetune_lr = 0.1
inc_start = 60
inc_end = 100
inc_step = 5

copy_list = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
step_list = (50, 50, 50, 50, 50, 50, 50, 50, 50, 50)
