_base_ = [
    '../_base_/models/obm_seg.py',
    '../_base_/datasets/bonai_instance_seg_building_roof_all.py',
    '../_base_/schedules/schedule_4x.py', 
    '../_base_/default_runtime.py'
]
load_from = '/config_data/work_dirs/obm_seg_b_coders_smallcoder/epoch_48.pth'

train_cfg = dict(
        max_num = 60,
)

model = dict(
    mask_decoder = dict(
        offset_aug=[dict(
            type='DeltaXYOffsetCoder_Transformer',
            image_size = (150,150),
            target_means=[0.0, 0.0],
            target_stds=[0.5, 0.5]), dict(
            type='DeltaXYOffsetCoder_Transformer',
            image_size = (300,300),
            target_means=[0.0, 0.0],
            target_stds=[0.5, 0.5]),],
        # hidden_dim = 1024,
    )
            
)