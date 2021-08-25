_base_ = 'yolov3_mvdarknet53_544_25e_db4_AC_sv.py'
load_from = None

# optimizer
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.0005, _delete_=True)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,  # same as burn-in in darknet
    warmup_ratio=0.1,
    step=[5, 20, 25])
# runtime settings
total_epochs = 30
