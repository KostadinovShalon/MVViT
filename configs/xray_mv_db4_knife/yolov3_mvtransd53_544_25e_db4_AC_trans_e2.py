_base_ = 'yolov3_mvtransd53v0_544_25e_db4_AC_sv.py'
# model settings

model = dict(
    backbone=dict(combination_block=4,
                  positional_encoding=True,
                  num_encoder_layers=2,
                  num_decoder_layers=1)
)

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=4,)