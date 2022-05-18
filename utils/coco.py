import shutil
import sys
sys.path.append(".")

import json
import os
import cv2

from utils.toSquare import toSquare
def merge(base_path, folder1, folder2):
    with open(f"{base_path}/{folder1}/_annotations.coco.json") as f:
        data1 = json.load(f)
    with open(f"{base_path}/{folder2}/_annotations.coco.json") as f:
        data2 = json.load(f)

    max_image_id = 0
    for image in data1["images"]:
        max_image_id = max(max_image_id, image["id"])
        shutil.copy(f"{base_path}/{folder1}/{image['file_name']}", f"{base_path}/{folder1}_{image['file_name']}")
        image["file_name"] = f"{folder1}_{image['file_name']}"
        
    max_image_id += 1
    for image in data2["images"]:
        shutil.copy(f"{base_path}/{folder2}/{image['file_name']}", f"{base_path}/{folder2}_{image['file_name']}")
        image["file_name"] = f"{folder2}_{image['file_name']}"
        image["id"] += max_image_id
    
    max_ann_id = 0
    for annotation in data1["annotations"]:
        max_ann_id = max(max_ann_id, annotation["id"])
        
    max_ann_id += 1
    for annotation in data2["annotations"]:
        annotation["image_id"] += max_image_id
        annotation["id"] += max_ann_id

        
    data1["annotations"] += data2["annotations"]
    data1["images"] += data2["images"]
    
    with open(f"{base_path}/_annotations.coco.json", "w") as f:
        json.dump(data1, f)

def create_annotation(root_dir, base_json="_annotations.coco.json", labels="one_class"):
    with open(f"{root_dir}/{base_json}") as f:
        data = json.load(f)

    if labels == "one_class":
        classes = ["tile"]
    elif labels == "colors":
        classes = ["red", "orange", "black", "blue"]
    elif labels == "values":
        classes = [str(i) for i in range(1, 14)] + ["j"]
    elif labels == "values+colors":
        classes = []
        classes += [str(i) for i in range(1, 14)] + ["j"]
        classes += ["red", "orange", "black", "blue"]
    elif labels == "10xvalues+colors":
        classes = []
        classes += ["red", "orange", "black", "blue"]
        classes += [str(i) for i in range(1, 14)] + ["j"]

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

    id = 1
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
        elif labels == "values+colors":
            a["category_id"] = to_id[val]
            a["id"] = id
            id += 1
            new_data["annotations"].append(a.copy())
            
            a["category_id"] = to_id[col]
            a["id"] = id
            id += 1
            new_data["annotations"].append(a.copy())
        elif labels == "10xvalues+colors":
            a["category_id"] = 10 * to_id[val] + to_id[col]
            a["id"] = id
            id += 1
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

def crop(base_path, json_name, save_dir):
    with open(f"{base_path}/{json_name}") as f:
        data = json.load(f)

    id_to_image = {}
    for image in data["images"]:
        img = cv2.imread(f"{base_path}/{image['file_name']}")
        id_to_image[image["id"]] = img
    
    id_to_name = {}
    for a in data["categories"]:
        id_to_name[a["id"]] = a["name"]
        os.makedirs(f"{save_dir}/{a['name']}", exist_ok=True)
    
    for idx, ann in enumerate(data["annotations"]):
        x, y, w, h = ann["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        tile = id_to_image[ann["image_id"]][y:y+h, x:x+w]
        save_path = f"{save_dir}/{id_to_name[ann['category_id']]}/{idx+1000}.png"
        cv2.imwrite(save_path, tile)

def fixBoxes(path, sz=640):
    with open(f"{path}") as f:
        data = json.load(f)
    for annotation in data["annotations"]:
        x, y, w, h = annotation["bbox"]
        if x < 0: x = 0
        if y < 0: y = 0
        if x + w > sz: w = sz-x
        if y + h > sz: h = sz-y
        annotation["bbox"] = [x, y, w, h]
    with open(path, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    '''
    dict_keys(['info', 'licenses', 'categories', 'images', 'annotations'])
    {'id': 0, 'name': 'rummikub-tails-dataset', 'supercategory': 'none'}
    {'id': 0, 'license': 1, 'file_name': '286-photo_jpg.rf.069b7d50f62ab1cb520dde801e4bca39.jpg', 'height': 598, 'width': 810, 'date_captured': '2022-04-06T09:17:48+00:00'}
    {'id': 0, 'image_id': 0, 'category_id': 46, 'bbox': [335, 179, 34, 53], 'area': 1802, 'segmentation': [], 'iscrowd': 0}
    '''
    path = "images/real_and_stacked_640"
    base_json = "_annotations.coco.json"
    merge(path, "stacked", "real")
    # recategorize("images/coco-test/_annotations.coco.json",
    #              "images/coco-test-640/_annotations.coco.json")
    # path = "images/rummy-6/train/"
    # path = "images/generated/erasing/train/"
    # crop(path,  base_json, "images/tiles_val/")
    resize(path, 640)
    fixBoxes(f"{path}/{base_json}", 640)
    create_annotation(path, base_json, "10xvalues+colors")
    create_annotation(path, base_json, "values+colors")
    create_annotation(path, base_json, "colors")
    create_annotation(path, base_json, "values")
    create_annotation(path, base_json, "one_class")
