from pylabel import importer

# type_ = "colors"
# type_ = "values"
# type_ = "one_class"
type_ = "annotations"
# dataset = importer.ImportCoco(
#     path=f"images/coco-test/_{type_}.coco.json",
#     path_to_images="images/coco-test/"
# )
dataset = importer.ImportCoco(
    path=f"images/generated/ultimate/train/_{type_}.coco.json",
    path_to_images="images/generated/ultimate/train/"
)

print(f"Number of images: {dataset.analyze.num_images}")
print(f"Number of classes: {dataset.analyze.num_classes}")
print(f"Classes:{dataset.analyze.classes}")
print(f"Class counts:\n{dataset.analyze.class_counts}")
print(f"Path to annotations:\n{dataset.path_to_annotations}")

# dataset.export.ExportToYoloV5(f"images/yolo-test/{type_}/data")