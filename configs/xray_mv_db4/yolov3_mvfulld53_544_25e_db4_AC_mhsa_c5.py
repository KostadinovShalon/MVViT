_base_ = 'yolov3_mvfulld53_544_25e_db4_AC_sv.py'
# model settings

model = dict(
    backbone=dict(combination_block=5,
                  positional_encoding=True,
                  fusion="add_layernorm",
                  multi_head=True,
                  self_attention=True)
)