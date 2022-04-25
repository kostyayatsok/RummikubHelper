import sys
sys.path.append(".")

import json
import os
import cv2

from utils.toSquare import toSquare
def merge():
    with open("images/generated/coco-wide/train/_annotations.coco.json") as f:
        data = json.load(f)
    with open("images/generated/coco-wide/train/coco-big/_annotations.coco.json") as f:
        data_big = json.load(f)
    with open("images/generated/coco-wide/train/coco-more/_annotations.coco.json") as f:
        data_more = json.load(f)

    max_id = 0
    for image in data["images"]:
        max_id = max(max_id, image["id"])
    
    big_start_id = max_id + 1
    for image in data_big["images"]:
        image["id"] = big_start_id + image["id"]
        max_id = max(max_id, image["id"])
    
        image["file_name"] = "coco-big/" + image["file_name"]
        
        data["images"].append(image)
    
    more_start_id = max_id + 1
    for image in data_more["images"]:
        image["id"] = more_start_id + image["id"]
    
        image["file_name"] = "coco-more/" + image["file_name"]
    
        data["images"].append(image)
    
    max_id = 0
    for annotation in data["annotations"]:
        max_id = max(max_id, annotation["id"])
    shift = max_id + 1
    for annotation in data_big["annotations"]:
        annotation["image_id"] += big_start_id
        annotation["id"] += shift
        max_id = max(max_id, annotation["id"])
        data["annotations"].append(annotation)

    shift = max_id + 1    
    for annotation in data_more["annotations"]:
        annotation["image_id"] += more_start_id
        annotation["id"] += shift
        data["annotations"].append(annotation)
    
    with open("images/generated/coco-wide/train/_merged.coco.json", "w") as f:
        json.dump(data, f)

def create_annotation(root_dir, base_json="_annotations.coco.json", labels="one_class"):
    with open(f"{root_dir}/{base_json}") as f:
        data = json.load(f)

    if labels == "one_class":
        classes = ["tile"]
    elif labels == "colors":
        classes = ["red", "orange", "black", "blue"]
    elif labels == "values":
        classes = [str(i) for i in range(1, 14)] + ["j"]
        
    to_id = {c:i+1 for i, c in enumerate(classes)}
   
    new_data = {
        "categories": [{'id': 0, 'name': 'rummikub-tiles-dataset', 'supercategory': 'none'}] +\
                      [{'id': v, 'name': k, 'supercategory': 'rummikub-tiles-dataset'} for k, v in to_id.items()],
        "images": data["images"],
        "annotations": [],
        "classes": classes
    }

    old_to_name = {}
    for a in data["categories"]:
        old_to_name[a["id"]] = a["name"]

    for a in data["annotations"]:
        
        val, col = old_to_name[a["category_id"]].split("-")

        if labels=="one_class":
            a["category_id"] = 1
            new_data["annotations"].append(a.copy())
        elif labels == "values":
            a["category_id"] = to_id[val]
            new_data["annotations"].append(a.copy())    
        elif labels == "colors":
            a["category_id"] = to_id[col]
            new_data["annotations"].append(a.copy())

    with open(f"{root_dir}/_{labels}.coco.json", "w") as f:
        json.dump(new_data, f)

def resize(root_dir, size=640, base_json="_annotations.coco.json"):
    with open(f"{root_dir}/{base_json}") as f:
        data = json.load(f)

    for category in data["categories"]:
        category["name"] = category["name"].replace('_', '-')

    ratios = {}
    for image in data["images"]:
        img = cv2.imread(f"{root_dir}/{image['file_name']}")
        img = toSquare(img, size)
        cv2.imwrite(f"{root_dir}/{image['file_name']}", img)
        ratios[image["id"]] = size / max(image["height"], image["width"])
        image["height"]=size
        image["width"]=size

    for ann in data["annotations"]:
        ratio = ratios[ann["image_id"]]
        ann["bbox"] = [int(ann["bbox"][i]*ratio) for i in range(4)]
        ann["area"] = int(ann["area"]*ratio*ratio)
        
    with open(f"{root_dir}/{base_json}", "w") as f:
        json.dump(data, f)

def recategorize(source_json, reference_json):
    with open(source_json) as f:
        source = json.load(f)
    with open(reference_json) as f:
        reference = json.load(f)

    old_id_to_name = {}
    for category in source["categories"]:
        old_id_to_name[category["id"]] = category["name"].replace('_', '-')
    new_name_to_id = {}
    for category in reference["categories"]:
        new_name_to_id[category["name"]] = category["id"]
    for annotation in source["annotations"]:
        annotation["category_id"] = new_name_to_id[old_id_to_name[annotation["category_id"]]]
    source["categories"] = reference["categories"]

    with open(source_json, "w") as f:
        json.dump(source, f)

if __name__ == "__main__":
    '''
    dict_keys(['info', 'licenses', 'categories', 'images', 'annotations'])
    {'id': 0, 'name': 'rummikub-tails-dataset', 'supercategory': 'none'}
    {'id': 0, 'license': 1, 'file_name': '286-photo_jpg.rf.069b7d50f62ab1cb520dde801e4bca39.jpg', 'height': 598, 'width': 810, 'date_captured': '2022-04-06T09:17:48+00:00'}
    {'id': 0, 'image_id': 0, 'category_id': 46, 'bbox': [335, 179, 34, 53], 'area': 1802, 'segmentation': [], 'iscrowd': 0}
    '''
    # recategorize("images/coco-test/_annotations.coco.json",
    #              "images/coco-test-640/_annotations.coco.json")
    path = "images/coco-test-1280/"
    base_json = "_annotations.coco.json"
    resize(path, 1280)
    create_annotation(path, base_json, "colors")
    create_annotation(path, base_json, "values")
    create_annotation(path, base_json, "one_class")
