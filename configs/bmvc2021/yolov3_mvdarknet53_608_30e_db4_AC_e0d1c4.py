_base_ = 'yolov3_mvdarknet53_608_30e_db4_AC_sv.py'
# model settings
model = dict(
    backbone=dict(
        combination_block=4,
        num_encoder_layers=0,
        num_decoder_layers=1,
    )
)
