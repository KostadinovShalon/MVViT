_base_ = 'yolov3_mvdarknet53_544_25e_db4_ABCD_sv.py'
# model settings
model = dict(
    backbone=dict(views=2)
)

# Modify dataset related settings
data = dict(
    train=dict(
        ann_files=['data/db4/db4_train_A.json', 'data/db4/db4_train_C.json']),
    val=dict(
        ann_files=['data/db4/db4_test_A.json', 'data/db4/db4_test_C.json']),
    test=dict(
        ann_files=['data/db4/db4_test_A.json', 'data/db4/db4_test_C.json']))
