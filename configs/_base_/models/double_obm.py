encoder_embed_dim=768
encoder_depth=12
encoder_num_heads=12
encoder_global_attn_indexes=[2, 5, 8, 11]


prompt_embed_dim = 256
image_size = 1024
vit_patch_size = 16
image_embedding_size = image_size // vit_patch_size

model = dict(
    type='DoubleHead_OBM',
    image_encoder=dict(
            type = 'ImageEncoderViT',
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
    ),
    prompt_encoder = dict(
        type = 'PromptEncoder',
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    ),
    mask_decoder1 = dict(
        type = 'single_MaskDecoder_seg',
        transformer_dim=prompt_embed_dim,
        transformer = dict(
            type = 'TwoWayTransformer',
            embedding_dim=prompt_embed_dim,
            depth=2,
            mlp_dim=2048,
            num_heads=8,
        ),),
    mask_decoder2 = dict(
        type = 'single_MaskDecoder_seg',
        transformer_dim=prompt_embed_dim,
        transformer = dict(
            type = 'TwoWayTransformer',
            embedding_dim=prompt_embed_dim,
            depth=2,
            mlp_dim=2048,
            num_heads=8,
        ),)
)


# train_cfg = dict(
#     sampler = dict(
#         type='OffsetSampler',
#         max_num = 40,
#         t_len = 24, # lower the prompt whose length is shorter than t_len
#     ),  
#     )
train_cfg = dict(
    max_num = 40,
)

test_cfg = dict(
)
