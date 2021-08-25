_base_ = 'yolov3_d53_544_273e_db4_padcentre_AC_sv.py'
# model settings

model = dict(
    backbone=dict(epipolar_combination_block=4, k=32)
)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,)
lr_config = dict(step=[25])
