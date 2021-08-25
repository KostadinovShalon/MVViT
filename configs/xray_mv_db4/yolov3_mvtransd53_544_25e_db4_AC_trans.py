_base_ = 'yolov3_mvtransd53v0_544_25e_db4_AC_sv.py'
# model settings

model = dict(
    backbone=dict(combination_block=4,
                  positional_encoding=True)
)

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=4,)