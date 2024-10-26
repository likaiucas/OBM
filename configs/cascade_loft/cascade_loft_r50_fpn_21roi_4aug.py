
_base_ = [
    '../_base_/models/cascade_loft_r50_fpn.py',
    '../_base_/datasets/bonai_instance.py',
    '../_base_/schedules/schedule_2x_bonai.py', 
    '../_base_/default_runtime.py'
]
model = dict(
    roi_head=dict(
        offset_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=16, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
        offset_head=dict(
                type='OffsetHeadExpandFeature',
                roi_feat_size=16,
                expand_feature_num=4,
                share_expand_fc=True,
                rotations=[0, 90, 180, 270],
                num_fcs=2,
                fc_out_channels=1024,
                num_convs=10,
                loss_offset=dict(type='SmoothL1Loss', loss_weight=8*2.0)))
)