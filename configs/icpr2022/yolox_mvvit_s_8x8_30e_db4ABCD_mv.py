_base_ = 'yolox_mvvit_s_8x8_30e_db4ABCD_sv.py'
model = dict(
    backbone=dict(
        combination_block=4,
        num_decoder_layers=1,
    )
)