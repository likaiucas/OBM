model = dict(
        type='OBM',
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        # checkpoint='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
)
train_cfg = dict(
        max_num = 100,
)
test_cfg = dict(
)
