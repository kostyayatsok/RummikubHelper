# %%writefile {exp_name}_config.py

annotation_filename = "values+colors"
run_name = f"{annotation_filename}_yolox_s"
root_data_dir = "rummy-data"
classes = ('rummikub-tiles-dataset', "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "j", "red", "orange", "black", "blue", )
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
    samples_per_gpu=4,
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

log_config = dict(
    hooks=[
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='RummyMMD-ultimate',
                name=run_name
            ),
        ),
        dict(type='TextLoggerHook'),
    ])
