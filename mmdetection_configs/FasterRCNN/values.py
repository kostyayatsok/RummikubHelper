run_name = "values_fasterRCNN"
root_data_dir = "rummy-data"
annotation_filename = "values"
classes = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', )
num_classes = len(classes)


# The new config inherits a base config to highlight the necessary modification
_base_ = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes),
    )
)

# Modify dataset related settings
dataset_type = 'COCODataset'
data = dict(
    train=dict(
        img_prefix=f'{root_data_dir}/train/',
        classes=classes,
        ann_file=f'{root_data_dir}/train/_{annotation_filename}.coco.json'),
    val=dict(
        img_prefix=f'{root_data_dir}/val/',
        classes=classes,
        ann_file=f'{root_data_dir}/val/_{annotation_filename}.coco.json'),
    test=dict(
        img_prefix=f'coco-test/',
        classes=classes,
        ann_file=f'coco-test/_{annotation_filename}.coco.json')
)

optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

log_config = dict(
    interval=10,
    hooks=[
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='RummyMMD-ultimate',
                name=run_name
            )
        )
    ])
# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'