_base_ = 'faster_rcnn_swin-t-p4-w7_mvvit_fpn_ms-crop-3x_db4AC_sv.py'
model = dict(
    backbone=dict(
        combination_block=4,
        num_decoder_layers=1,
    )
)
