_base_ = 'yolov3_mvdarknet53_608_10e_wisenet45_sv.py'
# model settings
model = dict(
    backbone=dict(
        combination_block=4,
        num_decoder_layers=1,
        shared_transformer=True
    )
)
