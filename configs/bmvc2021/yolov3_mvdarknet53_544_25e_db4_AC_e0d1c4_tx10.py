_base_ = 'yolov3_mvdarknet53_544_25e_db4_AC_e0d1c4.py'
# model settings
optimizer = dict(paramwise_cfg=dict(custom_keys={
                            '.transformer': dict(lr_mult=10)})
                        )