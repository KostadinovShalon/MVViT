_base_ = 'yolov3_d53_544_273e_db4_AC_sv.py'
# model settings

fundamental_matrices = {(0, 1): [[5.21118846e-07, -1.75891157e-06, 1.21004892e-02],
                                 [3.73646135e-07, 5.13830925e-09, -1.56484969e-04],
                                 [-1.21000050e-02, 5.65813040e-04, -3.97609620e-02]]}
size = (544, 544)

# dataset settings
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadMVImagesFromFile', to_float32=True),
    dict(type='LoadMVAnnotations', with_bbox=True),
    dict(type='MVResize', img_scale=size, keep_ratio=True),
    dict(type='MVNormalize', **img_norm_cfg),
    dict(type='MVPad', size=size, pad_val=(1., 1., 1.), pad_to_centre=True),
    dict(type='MVFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'], meta_keys=('filename', 'ori_filename', 'ori_shape',
                                                                            'img_shape', 'pad_shape',
                                                                            'scale_factor','pad_to_centre',
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
            dict(type='MVPad', size=size, pad_val=(1., 1., 1.), pad_to_centre=True),
            dict(type='MVImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], meta_keys=('filename', 'ori_filename', 'ori_shape',
                                                          'img_shape', 'pad_shape','pad_to_centre',
                                                          'scale_factor', 'img_norm_cfg'))
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        pipeline=train_pipeline,
        fundamental_matrices=fundamental_matrices),
    val=dict(
        pipeline=test_pipeline,
        fundamental_matrices=fundamental_matrices),
    test=dict(
        pipeline=test_pipeline,
        fundamental_matrices=fundamental_matrices))

optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,  # same as burn-in in darknet
    warmup_ratio=0.1,
    step=[18, 22])
# runtime settings
total_epochs = 30
evaluation = dict(interval=1, metric=['bbox'])

custom_imports = dict(imports=['mv_extension.hooks.grad_hook'], allow_failed_imports=False)
custom_hooks = [
    dict(type='GradHook')
]