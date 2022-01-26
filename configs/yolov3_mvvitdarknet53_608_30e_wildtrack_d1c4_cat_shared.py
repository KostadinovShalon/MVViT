_base_ = 'yolov3_mvvitdarknet53_608_30e_wildtrack_sv.py'
model = dict(
    backbone=dict(
        combination_block=4,
        num_decoder_layers=1,
    )
)
