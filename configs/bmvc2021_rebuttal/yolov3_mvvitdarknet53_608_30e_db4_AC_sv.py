_base_ = 'yolov3_mvvitdarknet53_608_30e_db4_ABCD_sv.py'
# model settings
model = dict(
    backbone=dict(views=2)
)

# Modify dataset related settings
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        ann_files=['data/db4/db4_train_A.json', 'data/db4/db4_train_C.json']),
    val=dict(
        ann_files=['data/db4/db4_test_A.json', 'data/db4/db4_test_C.json']),
    test=dict(
        ann_files=['data/db4/db4_test_A.json', 'data/db4/db4_test_C.json']))