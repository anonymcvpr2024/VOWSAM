_base_ = ['./tracktor_mask-rcnn_r50_fpn_4e_mot17-private-half.py']

model = dict(
    detector=dict(
        rpn_head=dict(bbox_coder=dict(clip_border=True)),
        roi_head=dict(
            bbox_head=dict(bbox_coder=dict(clip_border=True), num_classes=1)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'work_dirs/mask_rcnn_lr_0.01_dist_4GPU_2BatchSize_resume/epoch_10.pth'
            # 'https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_8e_mot20-half_20210805_001244-2c323fd1.pth'  # noqa: E501
        )),
    reid=dict(
        head=dict(num_classes=1705),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'reid_r50_6e_mot20_20210803_212426-c83b1c01.pth'
            # 'https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot20_20210803_212426-c83b1c01.pth'  # noqa: E501
        )))
data_root = 'data/MOT20/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDetections'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1088, 1088),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img', 'public_bboxes'])
        ])
]
data = dict(
    train=dict(
        ann_file=data_root + 'annotations/half-train_cocoformat.json',
        detection_file=data_root + 'annotations/half-train_detections.pkl',
        img_prefix=data_root + 'train'),
    val=dict(
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        detection_file=data_root + 'annotations/half-val_detections.pkl',
        img_prefix=data_root + 'train',
        pipeline=test_pipeline),
    test=dict(
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        detection_file=data_root + 'annotations/half-val_detections.pkl',
        img_prefix=data_root + 'train',
        pipeline=test_pipeline))
# learning policy
lr_config = dict(step=[6])
# runtime settings
total_epochs = 8
