import cv2
import os
from tqdm import tqdm
import json
import numpy as np
import sys
sys.path.append('.')
from data_utils.synthetic_dataset import SyntheticConfig, SyntheticDataset
from math import floor

VERSION = "smooth"
PATH = f"images/generated/{VERSION}"
PART = "train"
# PART = "val"
IMG_SIZE = 416
FORMAT = "COCO"

if PART == 'val':
    N_SAMPLES = 50
else:
    N_SAMPLES = 2000


classes_names = []
for val in [str(i) for i in range(1, 14)] + ["j"]:
    for col in ["blue", "black", "orange", "red"]:
        classes_names.append(f"{val}-{col}")

# Create dict to map class name to index
name2class = {name:idx+1 for idx, name in enumerate(classes_names)}

synth_config = SyntheticConfig()
synth_config.dataset_size = N_SAMPLES
synth_config.image_size = IMG_SIZE
synth_dataset = SyntheticDataset(synth_config)

def generateSample(idx, labels_len):
    image, annotations = synth_dataset[idx]
    S = image.size(-1)
    labels = [{
        "id": labels_len+i,
        "image_id": idx,
        "category_id": name2class[synth_dataset.to_name[ann["label"]]],
        "bbox": [floor(ann["x0"]*S),floor(ann["y0"]*S),floor(ann["w"]*S),floor(ann["h"]*S)],
        "area": floor(ann["w"]*ann["h"]*S*S),
        "iscrowd": 0
    } for i, ann in enumerate(annotations)]
    image = (image.numpy().transpose(1, 2, 0) * 255.).astype(np.uint8)
    return image, labels

if __name__ == "__main__":
    images_coco = []
    labels_coco = []
    categories_coco = [
        {"id": 0, "name": "rummikub-tiles-dataset", "supercategory": None}] + [
        {"id": v, "name": k, "supercategory": "rummikub-tiles-dataset"} \
            for k, v in name2class.items()
    ]
    classes_coco = [a["name"] for a in categories_coco]
    
    os.makedirs(f"{PATH}/{PART}", exist_ok=True)
    
    for sample_id in tqdm(range(N_SAMPLES)):
        img, labels = generateSample(sample_id, len(labels_coco))
        cv2.imwrite(f"{PATH}/{PART}/{sample_id:06d}.jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        labels_coco.extend(labels)
        images_coco.append({
            "id": sample_id,
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            "file_name": f"{sample_id:06d}.jpg"
        })

    with open(f"{PATH}/{PART}/_annotations.coco.json", "w") as f:
        json.dump(
            {
                "categories": categories_coco,
                "images": images_coco,
                "annotations": labels_coco,
                "classes": classes_coco
            },
            f
        )