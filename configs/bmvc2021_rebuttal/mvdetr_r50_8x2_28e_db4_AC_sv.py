_base_ = 'mvdetr_r50_8x2_28e_db4_AC.py'
model = dict(bbox_head=dict(single_view=True))
