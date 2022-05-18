# %%writefile {exp_name}_config.py

run_name = "tiles_yolox_tiny"
root_data_dir = "rummy-data"
annotation_filename = "one_class"
classes = ('rummikub-tiles-dataset', 'tile',)
num_classes = len(classes)

_base_ = 'configs/yolox/yolox_tiny_8x8_300e_coco.py'
model = dict(
    bbox_head=dict(num_classes=num_classes),
)

dataset_type = 'CocoDataset'
train_dataset = dict(
    dataset=dict(
        ann_file=f'{root_data_dir}/train/_{annotation_filename}.coco.json',
        img_prefix=f'{root_data_dir}/train/',
        classes=classes,
    ),
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    persistent_workers=True,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=f'coco-test/_{annotation_filename}.coco.json',
        img_prefix=f'coco-test/'),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=f'coco-test/_{annotation_filename}.coco.json',
        img_prefix=f'coco-test/')
)

# optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

log_config = dict(
    hooks=[
        dict(
            type='WandbLoggerHook',
            # type='WandbLogger',
            # type='MMDetWandbHook',
            init_kwargs=dict(
                project='RummyMMD-ultimate',
                name=run_name
            ),
            # init_kwargs={
            #     'project': 'RummyMMD-ultimate',
            #     'name': run_name
            # },
            # interval=10,
            # log_checkpoint=True,
            # log_checkpoint_metadata=True,
            # num_eval_images=30
        ),
        dict(type='TextLoggerHook'),
    ])
# workflow = [('train', 1), ('val', 1)]
# interval = 2
# checkpoint_config = dict(interval=interval, max_keep_ckpts=10)

# restore = "work_dirs/{exp_name}_config/latest.pth"
# eval_config = dict(interval=1, save_best="val/bbox_mAP_50")
# checkpoint_config = dict(interval=1, max_keep_ckpts=10)
# load_from = 'checkpoints/yolov3_d53_320_273e_coco-421362b6.pth'
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'