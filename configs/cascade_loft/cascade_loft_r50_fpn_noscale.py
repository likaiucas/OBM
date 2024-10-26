
_base_ = [
    '../_base_/models/cascade_loft_r50_fpn.py',
    '../_base_/datasets/bonai_instance.py',
    '../_base_/schedules/schedule_2x_bonai.py', 
    '../_base_/default_runtime.py'
]
