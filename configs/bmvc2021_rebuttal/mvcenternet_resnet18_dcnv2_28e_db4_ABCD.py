_base_ = '../../../configs/centernet/centernet_resnet18_dcnv2_140e_coco.py'
# model settings

custom_imports = dict(imports=['MVViT.models.backbones.mvresnet',
                               'MVViT.models.dense_heads.mv_centernet_head',
                               'MVViT.models.necks.mv_ct_resnet_neck',
                               'MVViT.datasets.pipelines.formatting',
                               'MVViT.datasets.pipelines.loading',
                               'MVViT.datasets.pipelines.test_time_aug',
                               'MVViT.datasets.pipelines.transforms',
                               'MVViT.datasets.coco_mv',
                               'MVViT.datasets.custom_mv'], allow_failed_imports=False)

size = (512, 512)
classes = ('firearm', 'laptop', 'knife', 'camera')
model = dict(
    backbone=dict(type='MVResNet'),
    neck=dict(
        type='MVCTResNetNeck',
        views=4,
        multiview_decoder_mode='cat'),
    bbox_head=dict(
        type='MVCenterNetHead',
        num_classes=4))

dataset_type = 'MVCocoDataset'
checkpoint_config = dict(interval=1)
# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMVImagesFromFile', to_float32=True),
    dict(type='LoadMVAnnotations', with_bbox=True),
    dict(type='MVResize', img_scale=size, keep_ratio=True),
    dict(type='MVNormalize', **img_norm_cfg),
    dict(type='MVPad', size=size),
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
            dict(type='MVPad', size=size),
            dict(type='MVImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], meta_keys=('filename', 'ori_filename', 'ori_shape',
                                                          'img_shape', 'pad_shape',
                                                          'scale_factor', 'img_norm_cfg'))
        ])
]
# Modify dataset related settings
data = dict(
    samples_per_gpu=4,
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
optimizer = dict(lr=0.005)
evaluation = dict(interval=1, metric=['bbox'])

load_from = 'checkpoints/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth'
