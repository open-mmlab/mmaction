# model settings
model = dict(
    type='TSN2D',
    modality='Flow',
    in_channels=10,
    backbone=dict(
        type='BNInception',
        pretrained=None,
        bn_eval=False,
        partial_bn=True),
    spatial_temporal_module=dict(
        type='SimpleSpatialModule',
        spatial_type='avg',
        spatial_size=7),
    segmental_consensus=dict(
        type='SimpleConsensus',
        consensus_type='avg'),
    cls_head=dict(
        type='ClsHead',
        with_avg_pool=False,
        temporal_feature_size=1,
        spatial_feature_size=1,
        dropout_ratio=0.7,
        in_channels=1024,
        num_classes=101))
train_cfg = None
test_cfg = None
# dataset settings
dataset_type = 'RawFramesDataset'
data_root = 'data/ucf101/rawframes'
img_norm_cfg = dict(
    mean=[128], std=[1], to_rgb=False)
data = dict(
    test=dict(
        type=dataset_type,
        ann_file='data/ucf101/ucf101_val_split_1_rawframes.txt',
        img_prefix=data_root,
        img_norm_cfg=img_norm_cfg,
        num_segments=25,
        new_length=5,
        new_step=1,
        random_shift=False,
        modality='Flow',
        image_tmpl='flow_{}_{:05d}.jpg',
        img_scale=256,
        input_size=224,
        div_255=False,
        flip_ratio=0,
        resize_keep_ratio=True,
        oversample='ten_crop',
        random_crop=False,
        more_fix_crop=False,
        multiscale_crop=False,
        test_mode=True))

dist_params = dict(backend='nccl')
