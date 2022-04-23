# %%writefile {exp_name}_config.py

run_name = "all_stacked_yolov3"
root_data_dir = "rummy-data"
annotation_filename = "annotations"
classes = ("1-blue", "1-black", "1-orange", "1-red", "2-blue", "2-black", "2-orange", "2-red", "3-blue", "3-black", "3-orange", "3-red", "4-blue", "4-black", "4-orange", "4-red", "5-blue", "5-black", "5-orange", "5-red", "6-blue", "6-black", "6-orange", "6-red", "7-blue", "7-black", "7-orange", "7-red", "8-blue", "8-black", "8-orange", "8-red", "9-blue", "9-black", "9-orange", "9-red", "10-blue", "10-black", "10-orange", "10-red", "11-blue", "11-black", "11-orange", "11-red", "12-blue", "12-black", "12-orange", "12-red", "13-blue", "13-black", "13-orange", "13-red", "j-blue", "j-black", "j-orange", "j-red", )
num_classes = len(classes)

# The new config inherits a base config to highlight the necessary modification
_base_ = 'configs/yolo/yolov3_d53_320_273e_coco.py'
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
load_from = 'checkpoints/yolov3_d53_320_273e_coco-421362b6.pth'