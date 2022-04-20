import json
import os

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

if __name__ == "__main__":
    '''
    dict_keys(['info', 'licenses', 'categories', 'images', 'annotations'])
    {'id': 0, 'name': 'rummikub-tails-dataset', 'supercategory': 'none'}
    {'id': 0, 'license': 1, 'file_name': '286-photo_jpg.rf.069b7d50f62ab1cb520dde801e4bca39.jpg', 'height': 598, 'width': 810, 'date_captured': '2022-04-06T09:17:48+00:00'}
    {'id': 0, 'image_id': 0, 'category_id': 46, 'bbox': [335, 179, 34, 53], 'area': 1802, 'segmentation': [], 'iscrowd': 0}
    '''
    # merge()
    create_annotation("images/generated/ultimate/train", "_annotations.coco.json", "colors")
    create_annotation("images/generated/ultimate/train", "_annotations.coco.json", "values")
    create_annotation("images/generated/ultimate/train", "_annotations.coco.json", "one_class")
