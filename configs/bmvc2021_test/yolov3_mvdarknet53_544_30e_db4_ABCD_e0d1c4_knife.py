_base_ = '../bmvc2021/yolov3_mvdarknet53_544_25e_db4_ABCD_e0d1c4.py'

# Modify dataset related settings
data = dict(
    train=dict(
        ann_files=['data/db4/db4_train_A_knife.json', 'data/db4/db4_train_B_knife.json',
                   'data/db4/db4_train_C_knife.json', 'data/db4/db4_train_D_knife.json']),
    val=dict(
        ann_files=['data/db4/db4_test_A_knife.json', 'data/db4/db4_test_B_knife.json',
                   'data/db4/db4_test_C_knife.json', 'data/db4/db4_test_D_knife.json']),
    test=dict(
        ann_files=['data/db4/db4_test_A_knife.json', 'data/db4/db4_test_B_knife.json',
                   'data/db4/db4_test_C_knife.json', 'data/db4/db4_test_D_knife.json']))
