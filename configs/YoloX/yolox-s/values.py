# %%writefile {exp_name}_config.py

run_name = "values_yolox_s"
root_data_dir = "rummy-data"
annotation_filename = "values"
classes = ('rummikub-tiles-dataset', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', )
num_classes = len(classes)

_base_ = 'configs/yolox/yolox_s_8x8_300e_coco.py'
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'

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

# eval_config = dict(interval=1, save_best="val/bbox_mAP_50")
# checkpoint_config = dict(interval=1, max_keep_ckpts=10)
# load_from = 'checkpoints/yolov3_d53_320_273e_coco-421362b6.pth'