# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    # dict(
    #     type='LinearLR',  # 线性预热策略
    #     start_factor=1e-6,  # 起始因子（初始学习率为 base_lr * 1e-6）
    #     by_epoch=False,
    #     begin=0,
    #     end=2000),
    dict(
        type='PolyLR',    # 多项式衰减策略
        eta_min=1e-5,     # 最小学习率
        power=0.9,        # 衰减指数
        begin=0,          # 预热结束后开始
        end=160000,
        by_epoch=False)
]
# training schedule for 160k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=8000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000, max_keep_ckpts=10, save_best='mIoU', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
