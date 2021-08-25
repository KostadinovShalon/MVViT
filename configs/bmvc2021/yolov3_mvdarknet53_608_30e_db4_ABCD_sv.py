_base_ = 'yolov3_mvdarknet53_544_25e_db4_ABCD_sv.py'
# model settings

size = (608, 608)
model = dict(
    backbone=dict(input_size=size),
)

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
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        ann_files=['data/db4/db4_train_A.json', 'data/db4/db4_train_B.json',
                   'data/db4/db4_train_C.json', 'data/db4/db4_train_D.json'],
        pipeline=train_pipeline),
    val=dict(
        ann_files=['data/db4/db4_test_A.json', 'data/db4/db4_test_B.json',
                   'data/db4/db4_test_C.json', 'data/db4/db4_test_D.json'],
        pipeline=test_pipeline),
    test=dict(
        ann_files=['data/db4/db4_test_A.json', 'data/db4/db4_test_B.json',
                   'data/db4/db4_test_C.json', 'data/db4/db4_test_D.json'],
        pipeline=test_pipeline))
