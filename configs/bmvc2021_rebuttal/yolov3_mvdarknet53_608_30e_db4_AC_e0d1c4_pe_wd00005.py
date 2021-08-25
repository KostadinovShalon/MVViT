_base_ = 'yolov3_mvdarknet53_608_30e_db4_AC_e0d1c4_wd00005.py'
# model settings
model = dict(
    backbone=dict(positional_encoding=True
    )
)