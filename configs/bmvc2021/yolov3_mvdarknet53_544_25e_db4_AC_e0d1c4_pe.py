_base_ = 'yolov3_mvdarknet53_544_25e_db4_AC_e0d1c4.py'
# model settings
model = dict(
    backbone=dict(positional_encoding=True
    )
)
