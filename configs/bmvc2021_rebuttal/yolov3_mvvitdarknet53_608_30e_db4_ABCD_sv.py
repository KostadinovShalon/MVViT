_base_ = '../../../configs/yolo/yolov3_d53_mstrain-608_273e_coco.py'
# model settings

custom_imports = dict(imports=['MVViT.models.backbones.mvvit_darknet',
                               'MVViT.models.dense_heads.yolo_head',
                               'MVViT.datasets.pipelines.formatting',
                               'MVViT.datasets.pipelines.loading',
                               'MVViT.datasets.pipelines.test_time_aug',
                               'MVViT.datasets.pipelines.transforms',
                               'MVViT.datasets.coco_mv',
                               'MVViT.datasets.custom_mv'], allow_failed_imports=False)

size = (608, 608)
classes = ('firearm', 'laptop', 'knife', 'camera')
model = dict(
    backbone=dict(type='MVViTDarknet',
                  depth=53,
                  out_indices=(3, 4, 5),
                  combination_block=-2,
                  input_size=size,
                  views=4),
    bbox_head=dict(type='YOLOV3MVHead', num_classes=len(classes)),
)

dataset_type = 'MVCocoDataset'
checkpoint_config = dict(interval=1)
# dataset settings
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadMVImagesFromFile', to_float32=True),
    dict(type='LoadMVAnnotations', with_bbox=True),
    dict(type='MVResize', img_scale=size, keep_ratio=True),
    dict(type='MVNormalize', **img_norm_cfg),
    dict(type='MVPad', size=size, pad_val=(1., 1., 1.)),
    dict(type='MVFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'], meta_keys=('filename', 'ori_filename', 'ori_shape',
                                                                            'img_shape', 'pad_shape',
                                                                            'scale_factor',
                                                                            'img_norm_cfg'))
]
test_pipeline = [
    dict(type='LoadMVImagesFromFile', to_float32=True),
    dict(
        type='MVMultiScaleFlipAug',
        img_scale=size,
        flip=False,
        transforms=[
            dict(type='MVResize', keep_ratio=True),
            dict(type='MVNormalize', **img_norm_cfg),
            dict(type='MVPad', size=size, pad_val=(1., 1., 1.)),
            dict(type='MVImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], meta_keys=('filename', 'ori_filename', 'ori_shape',
                                                          'img_shape', 'pad_shape',
                                                          'scale_factor', 'img_norm_cfg'))
        ])
]
# Modify dataset related settings
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    train=dict(
        _delete_=True,
        type=dataset_type,
        img_prefix='data/db4/images/',
        classes=classes,
        ann_files=['data/db4/db4_train_A.json', 'data/db4/db4_train_B.json',
                   'data/db4/db4_train_C.json', 'data/db4/db4_train_D.json'],
        pipeline=train_pipeline),
    val=dict(
        _delete_=True,
        type=dataset_type,
        img_prefix='data/db4/images/',
        classes=classes,
        ann_files=['data/db4/db4_test_A.json', 'data/db4/db4_test_B.json',
                   'data/db4/db4_test_C.json', 'data/db4/db4_test_D.json'],
        pipeline=test_pipeline),
    test=dict(
        _delete_=True,
        type=dataset_type,
        img_prefix='data/db4/images/',
        classes=classes,
        ann_files=['data/db4/db4_test_A.json', 'data/db4/db4_test_B.json',
                   'data/db4/db4_test_C.json', 'data/db4/db4_test_D.json'],
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0005, _delete_=True)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,  # same as burn-in in darknet
    warmup_ratio=0.1,
    step=[20, 25])
# runtime settings
total_epochs = 30
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
evaluation = dict(interval=1, metric=['bbox'])

# custom_hooks = [dict(type='GradientCumulativeOptimizerHook')]
load_from = 'checkpoints/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth'
