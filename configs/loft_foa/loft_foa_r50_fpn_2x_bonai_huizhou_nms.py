
_base_ = [
    '../_base_/models/bonai_loft_foa_r50_fpn_basic.py',
    '../_base_/datasets/bonai_instance_huizhou.py',
    '../_base_/schedules/schedule_2x_bonai.py', 
    '../_base_/default_runtime.py'
]


test_cfg = dict(
    rcnn=dict(
        score_thr=0.5,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100,
        mask_thr_binary=0.5))