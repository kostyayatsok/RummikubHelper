# %%writefile {exp_name}_config.py

run_name = "values_stacked_yolov3_608"
root_data_dir = "rummy-data"
annotation_filename = "values"
classes = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', )
num_classes = len(classes)

# The new config inherits a base config to highlight the necessary modification
# _base_ = 'configs/yolo/yolov3_d53_320_273e_coco.py'
_base_ = 'configs/yolo/yolov3_d53_mstrain-608_273e_coco.py'
# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    bbox_head=dict(num_classes=num_classes),
)
# Modify dataset related settings
dataset_type = 'COCODataset'
data = dict(
    train=dict(
        img_prefix=f'{root_data_dir}/train/',
        classes=classes,
        ann_file=f'{root_data_dir}/train/_{annotation_filename}.coco.json'),
    # val=dict(
    #     img_prefix=f'{root_data_dir}/val/',
    #     classes=classes,
    #     ann_file=f'{root_data_dir}/val/_{annotation_filename}.coco.json'),
    val=dict(
        img_prefix=f'coco-test/',
        classes=classes,
        ann_file=f'coco-test/_{annotation_filename}.coco.json'),
    test=dict(
        img_prefix=f'coco-test/',
        classes=classes,
        ann_file=f'coco-test/_{annotation_filename}.coco.json')
)

# optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

log_config = dict(
    interval=10,
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
workflow = [('train', 1), ('val', 1)]

# eval_config = dict(interval=1, save_best="val/bbox_mAP_50")
checkpoint_config = dict(interval=1, max_keep_ckpts=10)
# load_from = 'checkpoints/yolov3_d53_320_273e_coco-421362b6.pth'
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth'