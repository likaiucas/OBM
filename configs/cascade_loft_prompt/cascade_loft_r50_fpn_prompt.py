
_base_ = [
    '../_base_/models/cascade_loft_r50_fpn.py',
    '../_base_/datasets/bonai_instance_prompt.py',
    '../_base_/schedules/schedule_2x_bonai.py', 
    '../_base_/default_runtime.py'
]
model =dict(
    type='CascadeLOFTprompt',
    roi_head=dict(
        type='CascadePromptHead',
        inference_repeat_tensor=100,
        inference_aug=False,)
)