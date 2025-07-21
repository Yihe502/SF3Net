_base_ = [
    '../_base_/datasets/vaihingen.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=(512, 512),
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SFFUNet_T',
        in_channels=3,
        out_indices=(0, 1, 2, 3, 4),
        mlp_ratio=4,
        pretrained='mmcls://mobilenet_v2',
        init_cfg=None),
    decode_head=dict(
        type='FSFDecodeHead_T',
        in_channels=[16, 32, 48, 96, 128],
        channels=16,
        dropout_ratio=0.1,
        num_classes=6,
        align_corners=False,
        loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                     dict(type='DiceLoss', loss_weight=3.0)]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

optimizer = dict(_delete_=True, type='AdamW', lr=0.00012, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=0.9, min_lr=0.0, by_epoch=False)


find_unused_parameters = True
max_iters = 80000
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=4000)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1.0e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='CosineAnnealingLR',
        begin=max_iters // 2,
        T_max=max_iters // 2,
        end=max_iters,
        by_epoch=False,
        eta_min=0)
]

train_dataloader=dict(batch_size=4)
