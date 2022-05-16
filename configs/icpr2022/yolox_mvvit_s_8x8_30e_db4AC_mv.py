_base_ = 'yolox_mvvit_s_8x8_30e_db4AC_sv.py'
model = dict(
    backbone=dict(
        combination_block=4,
        num_decoder_layers=1,
    )
)
resume_from = 'work_dirs/yolox_mvvit_s_8x8_30e_db4AC_mv/epoch_24.pth'

max_epochs = 30
num_last_epochs = 5
interval = 1

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]