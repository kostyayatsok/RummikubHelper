from collections import defaultdict
import json
import torch
from data_utils.utils import get_label_encoders
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Pad

class CocoFormatDataset(Dataset):
    def __init__(self, annotation_path, images_dir, img_sz=640):
        super().__init__()
        self.to_id, self.to_name = get_label_encoders()
        self.img_sz = img_sz


        with open(annotation_path) as f:
            annotation = json.load(f)
        old_id_to_name = {}
        for category in annotation["categories"]:
            old_id_to_name[category["id"]] = '-'.join(category["name"].split('_'))
        data = defaultdict(lambda: {
            "boxes": [],
            "labels": [],
            "area": [],
            "iscrowd": []
        })
        for ann in annotation["annotations"]:
            x, y, w, h = ann["bbox"]
            box = [x, y, x+w, y+h]
            label = self.to_id[old_id_to_name[ann["category_id"]]]

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

    def resize(self, image, targets):
        x = image
        c, h, w = x.shape
        padding = (0, 0, max(h, w)-w, max(h, w)-h)
        x = Pad(padding)(x)
        image = Resize(self.img_sz)(x)
        
        ratio = self.img_sz / max(h, w)
        targets["boxes"] = (targets["boxes"]*ratio).int()
        targets["area"] = (targets["area"]*ratio*ratio).int()
        
        return image, targets

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        image = read_image(self.data[idx]["file_path"]) / 255. 
        targets = {k:v for k, v in self.data[idx].items() if k != "file_path"}
        image, targets = self.resize(image, targets)
        return image, targets