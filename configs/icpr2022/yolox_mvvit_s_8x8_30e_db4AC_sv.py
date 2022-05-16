_base_ = '../../../configs/yolox/yolox_s_8x8_300e_coco.py'

img_scale = (640, 640)  # height, width
classes = ('firearm', 'laptop', 'knife', 'camera')
custom_imports = dict(imports=['MVViT.models.backbones.mvvit_csp_darknet',
                               'MVViT.models.dense_heads.mv_yolox_head',
                               'MVViT.models.detectors.mv_yolox',
                               'MVViT.datasets.pipelines.formatting',
                               'MVViT.datasets.pipelines.loading',
                               'MVViT.datasets.pipelines.test_time_aug',
                               'MVViT.datasets.pipelines.transforms',
                               'MVViT.datasets.coco_mv',
                               'MVViT.datasets.custom_mv'], allow_failed_imports=False)

# model settings
model = dict(
    type='MVYOLOX',
    backbone=dict(type='MVViTCSPDarknet',
                  combination_block=-1,
                  views=2),
    bbox_head=dict(
        type='MVYOLOXHead', num_classes=len(classes)))

# dataset settings
# data_root = 'data/coco/'
dataset_type = 'MVCocoDataset'
# dataset settings
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)

train_pipeline = [
    # dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(type='LoadMVImagesFromFile', to_float32=True),
    dict(type='LoadMVAnnotations', with_bbox=True),
    # dict(
    #     type='MVRandomAffine',
    #     scaling_ratio_range=(0.1, 2),
    #     border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    # dict(
    #     type='MixUp',
    #     img_scale=img_scale,
    #     ratio_range=(0.8, 1.6),
    #     pad_val=114.0),
    # dict(type='YOLOXHSVRandomAug'),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type='MVResize', img_scale=img_scale, keep_ratio=True),
    # dict(type='MVNormalize', **img_norm_cfg),
    dict(
        type='MVPad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(255.0, 255.0, 255.0))),
    # dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='MVFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'], meta_keys=('filename', 'ori_filename', 'ori_shape',
                                                                            'img_shape', 'pad_shape',
                                                                            'scale_factor',
                                                                            'img_norm_cfg'))
]

train_dataset = dict(
    _delete_=True,
    type=dataset_type,
    img_prefix='data/db4/images/',
    classes=classes,
    ann_files=['data/db4/db4_train_A.json', 'data/db4/db4_train_C.json'],
    pipeline=train_pipeline)

test_pipeline = [
    dict(type='LoadMVImagesFromFile', to_float32=True),
    dict(
        type='MVMultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='MVResize', keep_ratio=True),
            # dict(type='MVNormalize', **img_norm_cfg),
            # dict(type='RandomFlip'),
            dict(
                type='MVPad',
                pad_to_square=True,
                pad_val=dict(img=(255.0, 255.0, 255.0))),
            dict(type='MVImageToTensor', keys=['img']),
            # dict(type='MVFormatBundle'),
            dict(type='Collect', keys=['img'], meta_keys=('filename', 'ori_filename', 'ori_shape',
                                                          'img_shape', 'pad_shape',
                                                          'scale_factor'))
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    persistent_workers=True,
    train=train_dataset,
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

# optimizer
# default 8 gpu
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)

max_epochs = 30
num_last_epochs = 5
resume_from = None
interval = 1

# learning policy
lr_config = dict(
    _delete_=True,
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=3,  # 3 epoch
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)


custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]
checkpoint_config = dict(interval=interval)
evaluation = dict(
    save_best='auto',
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=interval,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
    metric='bbox')
log_config = dict(interval=50)
