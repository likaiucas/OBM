dataset_type = 'BONAI'
data_root = '../BONAI_data/BONAI-20230403T091731Z-002/BONAI/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', 
         with_bbox=True,
         with_mask=True,
         with_offset=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal', 'vertical']),
    # dict(type='Resize', img_scale=[(768, 768),(1280, 1280)], keep_ratio=True),
    # dict(type='Albu',
    #         bbox_params=dict(
    #         type='BboxParams',
    #         format='pascal_voc',
    #         label_fields=['gt_labels'],
    #         min_visibility=0.0,
    #         filter_lost_elements=True),
    #      transforms=[
    #         dict(
    #             type='RandomBrightnessContrast',
    #             brightness_limit=[0.1, 0.3],
    #             contrast_limit=[0.1, 0.3],
    #             p=0.2),
    #         dict(type='ChannelShuffle', p=0.1),
    #         dict(
    #             type='OneOf',
    #             transforms=[
    #                 dict(type='Blur', blur_limit=3, p=1.0),
    #                 dict(type='MedianBlur', blur_limit=3, p=1.0)
    #             ],
    #             p=0.1),
            
    #     ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_offsets']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
# cities = ['shanghai', 'beijing', 'haerbin', 'chengdu']

train_ann_file = []
img_prefix = []

val_ann_file = []
val_image_prefix = []
for city in cities:
    train_ann_file.append(data_root + 'coco/bonai_{}_trainval.json'.format(city))
    img_prefix.append("../BONAI_data/trainval-001/trainval/images")

val_ann_file.append(data_root + 'coco/bonai_jinan_trainval.json')
val_image_prefix.append("../BONAI_data/trainval-001/trainval/images")
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=train_ann_file,
        img_prefix=img_prefix,
        bbox_type='building',
        mask_type='roof',
        pipeline=train_pipeline),
    # val=dict(
    #     type=dataset_type,
    #     ann_file=val_ann_file[0],
    #     img_prefix=val_image_prefix[0],
    #     gt_footprint_csv_file="",
    #     pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='../BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco/bonai_shanghai_xian_test.json',
        img_prefix='../BONAI_data/BONAI-20230403T091731Z-002/BONAI/test/test',
        gt_footprint_csv_file="",
        bbox_type='building',
        mask_type='roof',
        pipeline=test_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='../BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco/bonai_shanghai_xian_test.json',
        img_prefix='../BONAI_data/BONAI-20230403T091731Z-002/BONAI/test/test',
        gt_footprint_csv_file="",
        bbox_type='building',
        mask_type='roof',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric=['bbox', 'segm'])