_base_ = '../../../configs/detr/detr_r50_8x2_150e_coco.py'

custom_imports = dict(imports=['MVViT.models.backbones.mvresnet',
                               'MVViT.models.dense_heads.mv_detr_head',
                               'MVViT.models.transformers.mv_positional_encoding',
                               'MVViT.models.transformers.mv_transformer',
                               'MVViT.datasets.pipelines.formatting',
                               'MVViT.datasets.pipelines.loading',
                               'MVViT.datasets.pipelines.test_time_aug',
                               'MVViT.datasets.pipelines.transforms',
                               'MVViT.datasets.coco_mv',
                               'MVViT.datasets.custom_mv'], allow_failed_imports=False)
classes = ('firearm', 'laptop', 'knife', 'camera')
model = dict(
    backbone=dict(type='MVResNet'),
    bbox_head=dict(
        type='MVDETRHead',
        num_classes=len(classes),
        transformer=dict(type='MVTransformer'),
        positional_encoding=dict(
            type='MVSinePositionalEncoding')),
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadMVImagesFromFile', to_float32=True),
    dict(type='LoadMVAnnotations', with_bbox=True),
    dict(
        type='MVResize',
        img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                   (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                   (736, 1333), (768, 1333), (800, 1333)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='MVNormalize', **img_norm_cfg),
    dict(type='MVPad', size_divisor=1, pad_val=(1., 1., 1.)),
    dict(type='DefaultFormatBundle'),
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
            dict(type='MVResize', keep_ratio=True),
            dict(type='MVNormalize', **img_norm_cfg),
            dict(type='MVPad', size_divisor=1, pad_val=(1., 1., 1.)),
            dict(type='MVImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], meta_keys=('filename', 'ori_filename', 'ori_shape',
                                                          'img_shape', 'pad_shape',
                                                          'scale_factor', 'img_norm_cfg'))
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[18, 24])
runner = dict(type='EpochBasedRunner', max_epochs=28)
