
_base_ = [
    '../_base_/models/bonai_loft_foa_r50_fpn_basic.py',
    '../_base_/datasets/bonai_instance.py',
    '../_base_/schedules/schedule_2x_bonai.py', 
    '../_base_/default_runtime.py'
]
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=3,
    train=dict(
        bbox_type='roof',
        mask_type='roof',),

    test=dict(
        bbox_type='roof',
        mask_type='roof',
        ),
    val=dict(
        bbox_type='roof',
        mask_type='roof',
))