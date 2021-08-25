_base_ = 'yolov3_mvvitdarknet53_608_30e_db4_AC_sv.py'
model = dict(
    backbone=dict(
        combination_block=4,
        num_decoder_layers=1,
        positional_encoding=False
    )
)