_base_ = 'yolov3_mvtransd53v2_544_25e_db4_ABCD_sv.py'
# model settings

model = dict(
    backbone=dict(combination_block=5,
                  positional_encoding=False,
                  num_encoder_layers=1,
                  num_decoder_layers=1,
                  decoder_mode='add',
                  n_blocks=2)
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,)
