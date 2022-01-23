_base_ = 'deformable_detr_r50_16x2_50e_db4_AC.py'

model = dict(
    backbone=dict(
        combination_block=3))