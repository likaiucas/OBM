model = dict(
        type='OBM',
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
    )
train_cfg = dict(
)
test_cfg = dict(
)