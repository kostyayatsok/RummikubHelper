import fiftyone as fo
import fiftyone.utils.coco as fouc

# The directory containing the source images
# data_path = "images/rummy-6/val"
# data_path = "images/generated/coco/val"
data_path = "images/coco-test-1280"
# data_path = "images/coco-test-640"
# data_path = "images/generated/ultimate/train"
# data_path = "images/generated/stacked/train"
# data_path = "images/generated/smooth/train/"

# The path to the COCO labels JSON file
# labels_path = f"{data_path}/_colors.coco.json"
labels_path = f"{data_path}/_annotations.coco.json"
# labels_path = f"{data_path}/_values.coco.json"
# labels_path = f"{data_path}/_values+colors.coco.json"
# labels_path = f"{data_path}/_one_class.coco.json"

# Import the dataset
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=data_path,
    labels_path=labels_path,
    include_id=True,
    # label_field="ground_truth",
)

if False:
    pred_label = "predictions"
    # classes=["red", "blue", "black", "orange"]
    classes=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
    # classes=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', 'j', "red", "blue", "black", "orange"]
    fouc.add_coco_labels(
        dataset,
        label_field=pred_label,
        # labels_or_path=f"predictions/values_yolov5l_predictions.bbox.json",
        # labels_or_path=f"predictions/tiles_stacked_yolov3_predictions.bbox.json",
        labels_or_path=f"predictions/values_yolox_l_predictions.bbox.json",
        # labels_or_path=f"predictions/values+colors_yolox_s_predictions.bbox.json",
        # labels_or_path=f"values_synth_fasterRCNN_predictions.bbox.json",
        # labels_or_path=f"colors_fasterRCNN_synthetic_predictions.bbox.json",
        # labels_or_path=f"colors_fasterRCNN_predictions.json",
        coco_id_field="coco_id",
    )

    results = dataset.evaluate_detections(
        pred_label,
        gt_field="detections",
        compute_mAP=True,
        # method="open-images",
        classwise=False,
    )

    results.print_report(classes=classes)
    print(results.mAP())
    # plot = results.plot_pr_curves(classes=classes)
    # plot.show()

    # plot = results.plot_roc_curves(classes=classes)
    # plot.show()

    # plot = results.plot_confusion_matrix(classes=classes)
    # plot.show()

session = fo.launch_app(dataset)

session.wait()
