_base_ = '../../../configs/yolo/yolov3_d53_320_273e_coco.py'
# model settings

custom_imports = dict(imports=['mv_extension.models.backbones.full_mv_transformer_darknet_v2'], allow_failed_imports=False)

size = (544, 544)
model = dict(
    backbone=dict(type='MVTransformerDarknet', depth=53, out_indices=(3, 4, 5), combination_block=-2,
                  input_size=size, views=2),
    bbox_head=dict(type='YOLOV3MVHead', num_classes=4),
)

dataset_type = 'MVCocoDataset'
checkpoint_config = dict(interval=3)
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
classes = ('firearm', 'laptop', 'knife', 'camera')
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=4,
    train=dict(
        _delete_=True,
        type=dataset_type,
        img_prefix='data/db4/images/',
        classes=classes,
        ann_files=['data/db4/db4_train_A_knife.json', 'data/db4/db4_train_C_knife.json'],
        pipeline=train_pipeline),
    val=dict(
        _delete_=True,
        type=dataset_type,
        img_prefix='data/db4/images/',
        classes=classes,
        ann_files=['data/db4/db4_test_A_knife.json', 'data/db4/db4_test_C_knife.json'],
        pipeline=test_pipeline),
    test=dict(
        _delete_=True,
        type=dataset_type,
        img_prefix='data/db4/images/',
        classes=classes,
        ann_files=['data/db4/db4_test_A_knife.json', 'data/db4/db4_test_C_knife.json'],
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,  # same as burn-in in darknet
    warmup_ratio=0.1,
    step=[16, 22])
# runtime settings
total_epochs = 25
evaluation = dict(interval=1, metric=['bbox'])
load_from = 'checkpoints/yolov3_d53_320_273e_coco-421362b6.pth'
