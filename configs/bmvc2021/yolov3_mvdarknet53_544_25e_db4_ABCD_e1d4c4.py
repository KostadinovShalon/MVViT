_base_ = 'yolov3_mvdarknet53_544_25e_db4_ABCD_sv.py'
# model settings
model = dict(
    backbone=dict(
        combination_block=4,
        num_encoder_layers=1,
        num_decoder_layers=4,
    )
)
