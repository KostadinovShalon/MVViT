_base_ = 'yolov3_mvtransd53_544_30e_db4_AC_sv.py'
# model settings

model = dict(
    backbone=dict(combination_block=4,
                  positional_encoding=True,
                  multi_head=True,
                  self_attention=True)
)