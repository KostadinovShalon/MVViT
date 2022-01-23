_base_ = [
    '../../../configs/_base_/datasets/coco_detection.py', '../../../configs/_base_/default_runtime.py'
]

custom_imports = dict(imports=['MVViT.models.backbones.mvvit_resnet',
                               'MVViT.models.necks.mv_channel_mapper',
                               'MVViT.models.dense_heads.mv_deformable_detr_head',
                               'MVViT.datasets.pipelines.formatting',
                               'MVViT.datasets.pipelines.loading',
                               'MVViT.datasets.pipelines.test_time_aug',
                               'MVViT.datasets.pipelines.transforms',
                               'MVViT.datasets.coco_mv',
                               'MVViT.datasets.custom_mv'], allow_failed_imports=False)

model = dict(
    type='DeformableDETR',
    backbone=dict(
        type='MVViTResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        views=7,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='MVChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='MVDeformableDETRHead',
        num_query=300,
        num_classes=1,
        in_channels=2048,
        sync_cls_avg_factor=True,
        as_two_stage=False,
        transformer=dict(
            type='DeformableDetrTransformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=256),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
    test_cfg=dict(max_per_img=100))
img_norm_cfg = dict(
    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadMVImagesFromFile', to_float32=True),
    dict(type='LoadMVAnnotations', with_bbox=True),
    dict(type='MVPad', size_divisor=1),
    dict(
        type='MVResize',
        img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                   (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                   (736, 1333), (768, 1333), (800, 1333)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='MVNormalize', **img_norm_cfg),
    dict(type='MVFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'], meta_keys=('filename', 'ori_filename', 'ori_shape',
                                                                            'img_shape', 'pad_shape',
                                                                            'scale_factor', 'img_norm_cfg'))
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type='LoadMVImagesFromFile', to_float32=True),
    dict(
        type='MVMultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='MVPad', size_divisor=1),
            dict(type='MVResize', keep_ratio=True),
            dict(type='MVNormalize', **img_norm_cfg),
            dict(type='MVImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], meta_keys=('filename', 'ori_filename', 'ori_shape',
                                                          'img_shape', 'pad_shape',
                                                          'scale_factor', 'img_norm_cfg'))
        ])
]
classes = ('person',)
dataset_type = 'MVCocoDataset'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type=dataset_type,
        img_prefix='data/Wildtrack/Image_subsets/',
        classes=classes,
        ann_files=['data/Wildtrack/view0_train.json',
                   'data/Wildtrack/view1_train.json',
                   'data/Wildtrack/view2_train.json',
                   'data/Wildtrack/view3_train.json',
                   'data/Wildtrack/view4_train.json',
                   'data/Wildtrack/view5_train.json',
                   'data/Wildtrack/view6_train.json'],
        pipeline=train_pipeline),
    val=dict(
        _delete_=True,
        type=dataset_type,
        img_prefix='data/Wildtrack/Image_subsets/',
        classes=classes,
        ann_files=['data/Wildtrack/view0_val.json',
                   'data/Wildtrack/view1_val.json',
                   'data/Wildtrack/view2_val.json',
                   'data/Wildtrack/view3_val.json',
                   'data/Wildtrack/view4_val.json',
                   'data/Wildtrack/view5_val.json',
                   'data/Wildtrack/view6_val.json'],
        pipeline=test_pipeline),
    test=dict(
        _delete_=True,
        type=dataset_type,
        img_prefix='data/Wildtrack/Image_subsets/',
        classes=classes,
        ann_files=['data/Wildtrack/view0_val.json',
                   'data/Wildtrack/view1_val.json',
                   'data/Wildtrack/view2_val.json',
                   'data/Wildtrack/view3_val.json',
                   'data/Wildtrack/view4_val.json',
                   'data/Wildtrack/view5_val.json',
                   'data/Wildtrack/view6_val.json'],
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[40])
runner = dict(type='EpochBasedRunner', max_epochs=50)
