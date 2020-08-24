# model settings
model = dict(
    type='SSN2D',
    backbone=dict(
        type='BNInception',
        pretrained='open-mmlab://bninception_caffe',
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
train_cfg = dict(
    ssn=dict(
        assigner=dict(
            fg_iou_thr=0.7,
            bg_iou_thr=0.01,
            incomplete_iou_thr=0.3,
            bg_coverage_thr=0.02,
            incomplete_overlap_thr=0.01),
        sampler=dict(
            num_per_video=8,
            fg_ratio=1,
            bg_ratio=1,
            incomplete_ratio=6,
            add_gt_as_proposals=True),
        loss_weight=dict(
            comp_loss_weight=0.1,
            reg_loss_weight=0.1),
        debug=False))
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
    videos_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file='data/thumos14/thumos14_tag_val_normalized_proposal_list.txt',
        img_prefix=data_root,
        img_norm_cfg=img_norm_cfg,
        train_cfg=train_cfg,
        test_cfg=test_cfg,
        input_format="NCHW",
        body_seg=model['segmental_consensus']['num_seg'][1],
        aug_seg=(model['segmental_consensus']['num_seg'][0],
                 model['segmental_consensus']['num_seg'][2]),
        aug_ratio=0.5,
        new_length=1,
        new_step=1,
        random_shift=True,
        modality='RGB',
        image_tmpl='img_{:05d}.jpg',
        img_scale=256,
        input_size=224,
        div_255=False,
        size_divisor=32,
        flip_ratio=0.5,
        resize_keep_ratio=True,
        test_mode=False,
        verbose=True),
    val=dict(
        type=dataset_type,
        ann_file='data/thumos14/thumos14_tag_test_normalized_proposal_list.txt',
        img_prefix=data_root,
        img_norm_cfg=img_norm_cfg,
        train_cfg=train_cfg,
        test_cfg=test_cfg,
        input_format="NCHW",
        body_seg=model['segmental_consensus']['num_seg'][1],
        aug_seg=(model['segmental_consensus']['num_seg'][0],
                 model['segmental_consensus']['num_seg'][2]),
        aug_ratio=0.5,
        new_length=1,
        new_step=1,
        random_shift=False,
        modality='RGB',
        image_tmpl='img_{:05d}.jpg',
        img_scale=256,
        input_size=224,
        div_255=False,
        size_divisor=32,
        flip_ratio=0,
        resize_keep_ratio=True,
        test_mode=False),
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
        img_scale=256,
        input_size=224,
        oversample=None,
        div_255=False,
        size_divisor=32,
        flip_ratio=0,
        resize_keep_ratio=True,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=1e-6)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    # warmup='linear',
    # warmup_iters=500,
    # warmup_ratio=1.0 / 3,
    step=[200, 400])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 450
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ssn_thumos14_2d_rgb_bn_inception'
load_from = None
resume_from = None
workflow = [('train', 1)]
