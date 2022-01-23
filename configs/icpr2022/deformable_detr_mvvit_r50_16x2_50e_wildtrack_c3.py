_base_ = 'deformable_detr_r50_16x2_50e_wildtrack.py'

model = dict(
    backbone=dict(
        combination_block=4))