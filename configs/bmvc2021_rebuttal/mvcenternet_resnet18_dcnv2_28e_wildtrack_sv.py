_base_ = 'mvcenternet_resnet18_dcnv2_28e_wildtrack'
# model settings
model = dict(neck=dict(single_view=True))
