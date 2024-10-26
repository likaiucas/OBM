
_base_ = [
    '../_base_/models/obm_seg.py',
    '../_base_/datasets/obm_bonai_instance_prompt.py',
    '../_base_/schedules/schedule_4x.py', 
    '../_base_/default_runtime.py'
]
load_from = '/config_data/copy_bonai/pretrained/sam_vit_b_01ec64.pth'
# load_from = 'work_dirs/loft_foa_r50_fpn_2x_bonai/latest.pth'
# resume_from = 'work_dirs/loft_foa_r50_fpn_2x_bonai_prompt_train_g4/epoch_8.pth'