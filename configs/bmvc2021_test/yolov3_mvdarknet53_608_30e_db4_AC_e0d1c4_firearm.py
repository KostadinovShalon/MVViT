_base_ = '../bmvc2021/yolov3_mvdarknet53_608_30e_db4_AC_e0d1c4.py'

# Modify dataset related settings
data = dict(
    train=dict(
        ann_files=['data/db4/db4_train_A_firearm.json', 'data/db4/db4_train_C_firearm.json']),
    val=dict(
        ann_files=['data/db4/db4_test_A_firearm.json', 'data/db4/db4_test_C_firearm.json']),
    test=dict(
        ann_files=['data/db4/db4_test_A_firearm.json', 'data/db4/db4_test_C_firearm.json']))
