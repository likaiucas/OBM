
_base_ = [
    './cascade_loft_r50_fpn_3x.py',

]
model=dict(
    pretrained='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth',
    backbone=dict(
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[ 6, 12, 24, 48 ],
        window_size=12,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=False
    ),
    neck=dict(in_channels=[128, 256, 512, 1024]),)