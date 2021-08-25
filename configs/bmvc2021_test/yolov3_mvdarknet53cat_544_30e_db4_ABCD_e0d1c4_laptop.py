_base_ = '../bmvc2021/yolov3_mvdarknet53cat_544_25e_db4_ABCD_e0d1c4.py'

# Modify dataset related settings
data = dict(
    train=dict(
        ann_files=['data/db4/db4_train_A_laptop.json', 'data/db4/db4_train_B_laptop.json',
                   'data/db4/db4_train_C_laptop.json', 'data/db4/db4_train_D_laptop.json']),
    val=dict(
        ann_files=['data/db4/db4_test_A_laptop.json', 'data/db4/db4_test_B_laptop.json',
                   'data/db4/db4_test_C_laptop.json', 'data/db4/db4_test_D_laptop.json']),
    test=dict(
        ann_files=['data/db4/db4_test_A_laptop.json', 'data/db4/db4_test_B_laptop.json',
                   'data/db4/db4_test_C_laptop.json', 'data/db4/db4_test_D_laptop.json']))