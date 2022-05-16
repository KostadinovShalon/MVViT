_base_ = [
    '../../../configs/_base_/models/faster_rcnn_r50_fpn.py',
    '../../../configs/_base_/datasets/coco_instance.py',
    '../../../configs/_base_/schedules/schedule_1x.py', '../../../configs/_base_/default_runtime.py'
]

# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

classes = ('firearm', 'laptop', 'knife', 'camera')
custom_imports = dict(imports=['MVViT.models.backbones.mvvit_swin',
                               'MVViT.models.dense_heads.mv_rpn_head',
                               'MVViT.models.roi_heads.mv_standard_roi_head',
                               'MVViT.datasets.pipelines.formatting',
                               'MVViT.datasets.pipelines.loading',
                               'MVViT.datasets.pipelines.test_time_aug',
                               'MVViT.datasets.pipelines.transforms',
                               'MVViT.datasets.coco_mv',
                               'MVViT.datasets.custom_mv'], allow_failed_imports=False)

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

model = dict(
    backbone=dict(
        _delete_=True,
        type='MVViTSwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained),
        combination_block=-1,
        views=2
    ),
    neck=dict(in_channels=[96, 192, 384, 768]),
    rpn_head=dict(
        type='MVRPNHead'
    ),
    roi_head=dict(
        type='MVStandardRoIHead',
        bbox_head=dict(
            num_classes=len(classes)
        )))

img_norm_cfg = dict(
    mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadMVImagesFromFile'),
    dict(type='LoadMVAnnotations', with_bbox=True, with_mask=False),
    dict(type='MVRandomFlip', flip_ratio=0.5),
    dict(type='MVPad', pad_to_square=True, pad_val=dict(img=(255.0, 255.0, 255.0))),
    dict(
        type='MVResize',
        img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                   (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                   (736, 1333), (768, 1333), (800, 1333)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='MVNormalize', **img_norm_cfg),
    dict(type='MVFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape',
                    'scale_factor', 'img_norm_cfg')),
]
test_pipeline = [
    dict(type='LoadMVImagesFromFile', to_float32=True),
    dict(
        type='MVMultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='MVPad', pad_to_square=True, pad_val=dict(img=(255.0, 255.0, 255.0))),
            dict(type='MVResize', keep_ratio=True),
            dict(type='MVNormalize', **img_norm_cfg),
            dict(type='MVImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], meta_keys=('filename', 'ori_filename', 'ori_shape',
                                                          'img_shape', 'pad_shape',
                                                          'scale_factor', 'img_norm_cfg'))
        ])
]
dataset_type = 'MVCocoDataset'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type=dataset_type,
        img_prefix='data/db4/images/',
        classes=classes,
        ann_files=['data/db4/db4_train_A.json', 'data/db4/db4_train_C.json'],
        pipeline=train_pipeline),
    val=dict(
        _delete_=True,
        type=dataset_type,
        img_prefix='data/db4/images/',
        classes=classes,
        ann_files=['data/db4/db4_test_A.json', 'data/db4/db4_test_C.json'],
        pipeline=test_pipeline),
    test=dict(
        _delete_=True,
        type=dataset_type,
        img_prefix='data/db4/images/',
        classes=classes,
        ann_files=['data/db4/db4_test_A.json', 'data/db4/db4_test_C.json'],
        pipeline=test_pipeline))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
lr_config = dict(warmup_iters=1000, step=[27, 33])
runner = dict(max_epochs=36)
evaluation = dict(metric=['bbox'])