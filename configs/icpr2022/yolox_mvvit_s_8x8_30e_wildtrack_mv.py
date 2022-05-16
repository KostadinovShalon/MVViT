_base_ = 'yolox_mvvit_s_8x8_30e_wildtrack_sv.py'
model = dict(
    backbone=dict(
        combination_block=4,
        num_decoder_layers=1,
    )
)
