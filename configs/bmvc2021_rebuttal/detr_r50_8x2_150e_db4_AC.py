_base_ = '../../../configs/detr/detr_r50_8x2_150e_coco.py'
model = dict(
    bbox_head=dict(
        num_classes=4,
        ))
classes = ('firearm', 'laptop', 'knife', 'camera')
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    train=dict(
        img_prefix='data/db4/images/',
        classes=classes,
        ann_file='data/db4/db4_train.json'),
    val=dict(
        img_prefix='data/db4/images/',
        classes=classes,
        ann_file='data/db4/db4_test.json'),
    test=dict(
        img_prefix='data/db4/images/',
        classes=classes,
        ann_file='data/db4/db4_test.json'))
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
load_from = 'checkpoints/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'
