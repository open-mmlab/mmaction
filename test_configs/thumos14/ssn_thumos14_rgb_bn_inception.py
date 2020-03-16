# model settings
model = dict(
    type='SSN2D',
    backbone=dict(
        type='BNInception',
        pretrained=None,
        bn_eval=False,
        partial_bn=True),
    spatial_temporal_module=dict(
        type='SimpleSpatialModule',
        spatial_type='avg',
        spatial_size=7),
    dropout_ratio=0.8,
    segmental_consensus=dict(
        type='StructuredTemporalPyramidPooling',
        standalong_classifier=True,
        stpp_cfg=(1, 1, 1),
        num_seg=(2, 5, 2)),
    cls_head=dict(
        type='SSNHead',
        dropout_ratio=0.,
        in_channels_activity=1024,
        in_channels_complete=3072,
        num_classes=20,
        with_bg=False,
        with_reg=True))
# model training and testing settings
test_cfg=dict(
    ssn=dict(
        sampler=dict(
            test_interval=6,
            batch_size=16),
        evaluater=dict(
            top_k=2000,
            nms=0.2,
            softmax_before_filter=True,
            cls_score_dict=None,
            cls_top_k=2)))
# dataset settings
dataset_type = 'SSNDataset'
data_root = './data/thumos14/rawframes/'
img_norm_cfg = dict(
    mean=[104, 117, 128], std=[1, 1, 1], to_rgb=False)
data = dict(
    test=dict(
        type=dataset_type,
        ann_file='data/thumos14/thumos14_tag_test_normalized_proposal_list.txt',
        img_prefix=data_root,
        img_norm_cfg=img_norm_cfg,
        train_cfg=train_cfg,
        test_cfg=test_cfg,
        input_format='NCHW',
        aug_ratio=0.5,
        new_length=1,
        new_step=1,
        random_shift=False,
        modality='RGB',
        image_tmpl='img_{:05d}.jpg',
        img_scale=(340, 256),
        input_size=224,
        oversample=None,
        div_255=False,
        size_divisor=32,
        flip_ratio=0,
        resize_keep_ratio=True,
        test_mode=True))

dist_params = dict(backend='nccl')
