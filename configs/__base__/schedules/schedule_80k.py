# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
# optimizer = dict(type='AdamW', lr=0.01, betas=(0.9, 0.999), weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

param_scheduler = [
    # dict(
    #     type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=800),
    dict(
        type='PolyLR',
        power=0.9,
        eta_min=1e-4,
        # begin=800,
        # end=80000,
        by_epoch=False,
    )
]

# learning policy
# param_scheduler = [
#     # dict(
#     #     type='LinearLR',   # 线性预热策略
#     #     start_factor=1e-6,   # 起始因子（初始学习率为 base_lr * 1e-6）
#     #     by_epoch=False,
#     #     begin=0,
#     #     end=2000),
#     dict(
#         type='PolyLR',
#         eta_min=1e-4,
#         power=0.9,
#         begin=0,
#         end=80000,
#         by_epoch=False)
# ]
# training schedule for 80k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000 * 2, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=400, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    # checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000, max_keep_ckpts=8, save_best='mIoU', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
