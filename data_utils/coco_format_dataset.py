from collections import defaultdict
import json
import torch
from utils.tools import get_label_encoders
from torchvision.io import read_image
from torch.utils.data import Dataset
from utils.tools import resize

class CocoFormatDataset(Dataset):
    def __init__(self, annotation_path, images_dir, img_sz=640, relabel=True):
        super().__init__()

        if relabel:
            self.to_id, self.to_name = get_label_encoders()
        self.img_sz = img_sz

        with open(annotation_path) as f:
            annotation = json.load(f)

        if relabel:    
            old_id_to_name = {}
            for category in annotation["categories"]:
                old_id_to_name[category["id"]] = '-'.join(category["name"].split('_'))
        if not relabel:
            self.to_name, self.to_id = {}, {}
            for category in annotation["categories"]:         
                self.to_name[category["id"]] = category["name"]
                self.to_id[category["name"]] = category["id"]
        
        data = defaultdict(lambda: {
            "boxes": [],
            "labels": [],
            "area": [],
            "iscrowd": []
        })
        for ann in annotation["annotations"]:
            x, y, w, h = ann["bbox"]
            box = [x, y, x+w, y+h]
            if relabel:
                label = self.to_id[old_id_to_name[ann["category_id"]]]
            else:
                label = ann["category_id"]
            data[ann["image_id"]]["boxes"].append(box)
            data[ann["image_id"]]["labels"].append(label)
            data[ann["image_id"]]["area"].append(w*h)
            data[ann["image_id"]]["iscrowd"].append(False)
        for image in annotation["images"]:
            data[image["id"]]["file_path"] = images_dir+"/"+image["file_name"]

        self.data = []
        for image_id, ann in data.items():
            self.data.append({
                "image_id" : torch.tensor([image_id]),
                "boxes"    : torch.tensor(ann["boxes"]),
                "labels"   : torch.tensor(ann["labels"]),
                "area"     : torch.tensor(ann["area"]),
                "iscrowd"  : torch.tensor(ann["iscrowd"]),
                "file_path": ann["file_path"]
            })

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        image = read_image(self.data[idx]["file_path"]) / 255. 
        targets = {k:v for k, v in self.data[idx].items() if k != "file_path"}
        image, targets = resize(image, targets, self.img_sz)
        return image, targets