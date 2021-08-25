_base_ = 'yolov3_mvadarknet53_544_25e_db4_AC_sv.py'
# model settings
model = dict(
    backbone=dict(
        combination_block=4,
    )
)