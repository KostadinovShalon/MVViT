_base_ = 'yolov3_mvfulld53_544_25e_db4_AB_sv.py'
# model settings

model = dict(
    backbone=dict(combination_block=4, positional_encoding=True)
)