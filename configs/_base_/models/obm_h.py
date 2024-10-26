model = dict(
        type='OBM',
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
    )
train_cfg = dict(
)
test_cfg = dict(
)