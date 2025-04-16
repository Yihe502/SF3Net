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
model_cfgs = dict(
    cfg=[
        # k,  t,  c, s  k:kernel size, t:expand ratio, c:output channels, s:stride
        [3,   1,  16, 1], # 1/2        0.464K  17.461M
        [3,   4,  32, 2], # 1/4 1      3.44K   64.878M
        [3,   3,  32, 1], #            4.44K   41.772M
        [5,   3,  64, 2], # 1/8 3      6.776K  29.146M
        [5,   3,  64, 1], #            13.16K  30.952M
        [3,   3,  128, 2], # 1/16 5     16.12K  18.369M
        [3,   3,  128, 1], #            41.68K  24.508M
        [5,   6,  160, 2], # 1/32 7     0.129M  36.385M
        [5,   6,  160, 1], #            0.335M  49.298M
        [3,   6,  160, 1], #            0.335M  49.298M
    ],
    channels=[32, 64, 128, 160],
    out_channels=[None, 256, 256, 256],
    embed_out_indice=[2, 4, 6, 9],
    decode_out_indices=[1, 2, 3],
    num_heads=8,
    c2t_stride=2,
)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='CaSaFormer',
        cfgs=model_cfgs['cfg'],
        channels=model_cfgs['channels'],
        out_channels=model_cfgs['out_channels'],
        embed_out_indice=model_cfgs['embed_out_indice'],
        decode_out_indices=model_cfgs['decode_out_indices'],
        depths=4,
        num_heads=model_cfgs['num_heads'],
        c2t_stride=model_cfgs['c2t_stride'],
        drop_path_rate=0.1,
        norm_cfg=norm_cfg,
    ),
    decode_head=dict(
        type='SimpleHead',
        in_channels=[256, 256, 256],
        in_index=[0, 1, 2],
        channels=256,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_weight=1.0),
            dict(type='DiceLoss', loss_weight=0.4)]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))