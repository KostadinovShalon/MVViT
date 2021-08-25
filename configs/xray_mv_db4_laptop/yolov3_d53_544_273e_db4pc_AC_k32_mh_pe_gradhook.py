_base_ = 'yolov3_d53_544_273e_db4pc_AC_k32_gradhook.py'
# model settings

model = dict(
    backbone=dict(multi_head=True, h=4, positional_encoding=True, fusion="add_layernorm")
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)
lr_config = dict(
    step=[20, 24])