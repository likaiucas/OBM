
_base_ = [
    '../_base_/models/cascade_loft_swin_fpn.py',
    '../_base_/datasets/bonai_instance.py',
    '../_base_/schedules/schedule_2x_bonai.py', 
    '../_base_/default_runtime.py'
]
model=dict(
    # pretrained='https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth',
    pretrained='pretrained/swin_tiny_patch4_window7_224_22k.pth',
    backbone=dict(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        ape=False,
        drop_path_rate=0.1,
        patch_norm=True,
        use_checkpoint=False
    ),
    neck=dict(in_channels=[96, 192, 384, 768]),)