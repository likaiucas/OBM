_base_ = [
    '../_base_/models/obm_seg.py',
    '../_base_/datasets/bonai_instance_seg_building_roof_all.py',
    '../_base_/schedules/schedule_4x.py', 
    '../_base_/default_runtime.py'
]

load_from = 'pretrained/obm_long.pth'

train_cfg = dict(
        max_num = 60,
)

model = dict(
    noise_box=50,
    mask_decoder = dict(
        offset_aug=[dict(
            type='DeltaXYOffsetCoder_Transformer',
            image_size = (150,150),
            target_means=[0.0, 0.0],
            target_stds=[0.5, 0.5]), 
                    dict(
            type='DeltaXYOffsetCoder_Transformer',
            image_size = (300,300),
            target_means=[0.0, 0.0],
            target_stds=[0.5, 0.5]),
        dict(
            type='DeltaXYOffsetCoder_Transformer',
            image_size = (400,400),
            target_means=[0.0, 0.0],
            target_stds=[0.5, 0.5]),
            ],
        # hidden_dim = 1024,
    )
            
)