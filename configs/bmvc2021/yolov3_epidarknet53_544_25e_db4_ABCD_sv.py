_base_ = '../../../configs/yolo/yolov3_d53_320_273e_coco.py'
# model settings

custom_imports = dict(imports=['mv_extension.models.backbones.epipolar_mv_attention_darknet',
                               'mv_extension.models.dense_heads.yolo_head',
                               'mv_extension.datasets.pipelines.formatting',
                               'mv_extension.datasets.pipelines.loading',
                               'mv_extension.datasets.pipelines.test_time_aug',
                               'mv_extension.datasets.pipelines.transforms',
                               'mv_extension.datasets.coco_mv',
                               'mv_extension.datasets.custom_mv'], allow_failed_imports=False)

size = (544, 544)
fundamental_matrices = {(0, 1): [[1.16170465e-07, 2.61469108e-08, -1.04499386e-02],
                                 [4.13383830e-08, 1.93810573e-11, 1.90466076e-05],
                                 [1.03756304e-02, 1.67451270e-05, -1.96387639e-03]],
                        (0, 2): [[4.45897814e-07, -2.79773902e-07, -1.15397430e-02],
                                 [-1.93436595e-06, 1.70730416e-08, 7.01132265e-04],
                                 [1.19589770e-02, -5.06732925e-06, -1.89748220e-01]],
                        (0, 3): [[8.61084607e-07, 4.09433153e-06, -1.26337751e-02],
                                 [-3.84213659e-06, 1.37648646e-08, 1.22134548e-03],
                                 [1.20479942e-02, -1.34729705e-03, 1.04143131e-01]],
                        (1, 2): [[-1.79943362e-07, -2.10977239e-06, -1.07863013e-02],
                                 [3.79047680e-07, 1.44431441e-08, -3.90174644e-05],
                                 [1.12371778e-02, 6.25744290e-04, -1.41609102e-01]],
                        (1, 3): [[2.90914088e-08, -3.86175658e-06, -9.40982606e-03],
                                 [3.62842154e-06, -2.58880198e-08, -7.68587125e-04],
                                 [9.36261615e-03, 7.55249150e-04, 1.86312272e-02]],
                        (2, 3): [[-9.39198544e-07, -5.85822811e-07, 1.16929267e-02],
                                 [2.17221017e-06, -6.09004283e-08, -8.70022430e-04],
                                 [-1.14421777e-02, 5.71255092e-04, -4.73356409e-02]]}
model = dict(
    backbone=dict(type='MVTransformerDarknet',
                  depth=53,
                  out_indices=(3, 4, 5),
                  combination_block=-2,
                  input_size=size,
                  views=4,
                  fundamental_matrices=fundamental_matrices),
    bbox_head=dict(type='YOLOV3MVHead', num_classes=4),
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
classes = ('firearm', 'laptop', 'knife', 'camera')
data = dict(
    samples_per_gpu=3,
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
evaluation = dict(interval=1, metric=['bbox'])
load_from = 'checkpoints/yolov3_d53_320_273e_coco-421362b6.pth'
