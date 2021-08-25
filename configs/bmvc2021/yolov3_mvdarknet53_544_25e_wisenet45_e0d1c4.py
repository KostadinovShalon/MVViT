_base_ = 'yolov3_mvdarknet53_544_25e_wisenet45_sv.py'
# model settings
model = dict(
    backbone=dict(
        combination_block=4,
        num_encoder_layers=0,
        num_decoder_layers=1,
    )
)
