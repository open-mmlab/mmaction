# model settings
model = dict(
    type='TSN3D',
    backbone=dict(
        type='C3D',
        pretrained='https://open-mmlab.s3.ap-northeast-2.amazonaws.com/pretrain/third_party/c3d_caffe_sports1m_pretrain-c8182401.pth'),
    spatial_temporal_module=dict(
        type='SimpleSpatialTemporalModule',
        spatial_type='identity',
        temporal_size=1,
        spatial_size=1),
    segmental_consensus=dict(
        type='SimpleConsensus',
        consensus_type='avg'),
    cls_head=dict(
        type='ClsHead',
        with_avg_pool=False,
        temporal_feature_size=1,
        spatial_feature_size=1,
        dropout_ratio=0.5,
        in_channels=4096,
        num_classes=101))
train_cfg = None
test_cfg = None
# dataset settings
dataset_type = 'RawFramesDataset'
data_root = 'data/ucf101/rawframes/'
data_root_val = 'data/ucf101/rawframes/'
img_norm_cfg = dict(
    mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False)
pre_mean_npy = 'configs/C3D/c3d_train01_16_128_171_mean.npy'
# ported from https://github.com/facebookarchive/C3D/blob/master/C3D-v1.0/examples/c3d_finetuning/train01_16_128_171_mean.binaryproto
data = dict(
    videos_per_gpu=30,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file='data/ucf101/ucf101_train_split_1_rawframes.txt',
        img_prefix=data_root,
        img_norm_cfg=img_norm_cfg,
        pre_mean_npy=pre_mean_npy,
        input_format="NCTHW",
        num_segments=1,
        new_length=16,
        new_step=1,
        random_shift=True,
        modality='RGB',
        image_tmpl='img_{:05d}.jpg',
        img_scale=(171, 128),
        input_size=112,
        div_255=False,
        flip_ratio=0.5,
        resize_keep_ratio=False,
        oversample=None,
        random_crop=False,
        more_fix_crop=False,
        multiscale_crop=True,
        scales=[1, 0.8],
        max_distort=0,
        test_mode=False),
    val=dict(
        type=dataset_type,
        ann_file='data/ucf101/ucf101_val_split_1_rawframes.txt',
        img_prefix=data_root_val,
        img_norm_cfg=img_norm_cfg,
        pre_mean_npy=pre_mean_npy,
        input_format="NCTHW",
        num_segments=1,
        new_length=16,
        new_step=1,
        random_shift=True,
        modality='RGB',
        image_tmpl='img_{:05d}.jpg',
        img_scale=(171,128),
        input_size=112,
        div_255=False,
        flip_ratio=0,
        resize_keep_ratio=False,
        oversample=None,
        random_crop=False,
        more_fix_crop=False,
        multiscale_crop=False,
        test_mode=False),
    test=dict(
        type=dataset_type,
        ann_file='data/ucf101/ucf101_val_split_1_rawframes.txt',
        img_prefix=data_root_val,
        img_norm_cfg=img_norm_cfg,
        pre_mean_npy=pre_mean_npy,
        input_format="NCTHW",
        num_segments=10,
        new_length=16,
        new_step=1,
        random_shift=True,
        modality='RGB',
        image_tmpl='img_{:05d}.jpg',
        img_scale=(171,128),
        input_size=112,
        div_255=False,
        flip_ratio=0,
        resize_keep_ratio=False,
        oversample=None,
        random_crop=False,
        more_fix_crop=False,
        multiscale_crop=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[20, 40])
checkpoint_config = dict(interval=5)
# workflow = [('train', 5), ('val', 1)]
workflow = [('train', 1)]
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 60
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/c3d_ucf101_3d_rgb_vgg_c3d_seg1_f16s1_b30_g8_sports1m'
load_from = None
resume_from = None



