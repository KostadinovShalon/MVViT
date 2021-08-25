_base_ = 'yolov3_mvtransd53_544_25e_db4_ABCD_sv.py'
# model settings

model = dict(
    backbone=dict(combination_block=4,
                  positional_encoding=True,
                  num_encoder_layers=1,
                  num_decoder_layers=1)
)

optimizer = dict(type='SGD', lr=0.00005, momentum=0.9, weight_decay=0.00005)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,)