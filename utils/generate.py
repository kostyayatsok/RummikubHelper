from __future__ import annotations
from xml.etree.ElementTree import VERSION
import cv2
import glob
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from tqdm import tqdm
import json

VERSION = "coco-big"
PATH = f"images/generated/{VERSION}"
PART = "train"
# PART = "val"
MAX_ON_ONE = 1
MIN_ON_ONE = 1
IMG_SIZE = 640
FORMAT = "COCO"

if PART == 'val':
    N_SAMPLES = 50
else:
    N_SAMPLES = 10000

# Create list of classes names
classes_names = [str(i) for i in range(1, 14)] +\
                ["j", "blue", "black", "orange", "red"]

classes_names = []
for val in [str(i) for i in range(1, 14)] + ["j"]:
    for col in ["blue", "black", "orange", "red"]:
        classes_names.append(f"{val}_{col}")

# Create dict to map class name to index
name2class = {name:idx for idx, name in enumerate(classes_names)}

# Load tiles and backgrounds images
tiles = glob.glob('images/tiles/**/*.png')
backgrounds = glob.glob('images/backgrounds/*.jpg')

n_tiles = len(tiles)

# Define tiles transformations
tile_aug = nn.Sequential(
    T.RandomAffine(degrees=360, scale=(0.2, 0.5)),
    # T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2, hue=0),
    # T.RandomPerspective(distortion_scale=0.8, p=1)
)

# Define composition (whole image) transformations
aug = nn.Sequential(
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2, hue=0),
    T.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))
)


def generateSample(sample_id, labels_len) -> np.array:
    # Define amount of tiles on sample
    # we'll try to put tiles uniformly
    # so it would be nRows rows of tiles and nCols columns of tiles
    nRows = 2
    nCols = 2
    n = nRows * nCols
    # Generate positions to put tiles relative image size.
    positions = np.zeros((nRows, nCols, 2))
    for i in range(nRows):
        for j in range(nCols):
            positions[i, j, 0] = 0.5 / nRows + i / nRows
            positions[i, j, 1] = 0.5 / nCols + j / nCols
    # Define tiles
    idxs = np.random.randint(0, n_tiles, n)
    
    # Load background
    result = cv2.imread(np.random.choice(backgrounds))
    bH, bW = result.shape[0], result.shape[1]
    # Pad background to square
    result = np.pad(
        result, ((0, max(0, bW - bH)), (0, max(0, bH - bW)), (0, 0))
    )
    bSize = result.shape[0]
    
    # Convert positions from relative to absolute
    positions = positions.reshape(-1, 2)
    positions[:, 0] *= bH
    positions[:, 1] *= bW
    positions = positions.astype(int)


    labels = []

    for idx, [y0, x0] in zip(idxs, positions):
        # Read tile image
        tile = cv2.imread(tiles[idx])
        # Parse tile type
        val, col = tiles[idx].split('/')[-2].split('-')

        # Convert image to torch tensor
        tile = torch.from_numpy(tile.transpose(2, 0, 1))
        # Transform tile
        tile = tile_aug(tile)
        # Convert torch tensor to numpy array
        tile = tile.numpy().transpose(1, 2, 0)

        # Find minimum spanning bounding box (because of scaling)
        x, y, w, h = cv2.boundingRect(cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY))
        tile = tile[y:y+h, x:x+w, :]
        
        # Mask for tile without black background
        mask = (np.sum(tile, axis=-1) != 0)
        
        # If tile doesn't fit on backround let's skip it
        if y0 + h > bH or x0 + w > bW:
            continue
        
        # Put tile on background
        result[y0:y0+h, x0:x0+w, :][mask] = tile[mask]

        if FORMAT == "Yolo":
            # Each tile has two classes -- value and color
            labels.append(
                [name2class[col], (x0+w/2)/bSize, (y0+h/2)/bSize, w/bSize, h/bSize]
            )
            labels.append(
                [name2class[val], (x0+w/2)/bSize, (y0+h/2)/bSize, w/bSize, h/bSize]
            )
        elif FORMAT == "COCO":
            labels.append({
                "id": labels_len,
                "image_id": sample_id,
                "category_id": name2class[f"{val}_{col}"]+1,
                "bbox": [int(x0),int(y0),int(w),int(h)],
                "area": int(w*h),
                "iscrowd": 0
            })
            labels_len += 1
    # Convert to torch tensor
    result = torch.from_numpy(result.transpose(2, 0, 1))
    # Transform whole image
    result = aug(result)
    # Convert torch tensor back to numpy array
    result = result.numpy().transpose(1, 2, 0)

    if FORMAT == "Yolo":
        # Resize each sample to one size
        result = cv2.resize(result, (IMG_SIZE, IMG_SIZE))
        
    return result, labels

if __name__ == "__main__":
    print(f"Choosing from {n_tiles} tiles.")
    
    if FORMAT == "YOLO":
        # Create data.yml with information about train/val location,
        # number of classes (nc) and class indecies to class names mapping 
        with open(f"{PATH}/data.yaml", "w") as data:
            data.write(f"train: {VERSION}/train\n")
            data.write(f"val: {VERSION}/val\n")
            data.write(f"nc: {len(classes_names)}\n")
            data.write(f"names: {str(classes_names)}\n")

        # Create dirictories
        os.makedirs(f"{PATH}/{PART}/labels", exist_ok=True)
        os.makedirs(f"{PATH}/{PART}/images", exist_ok=True)
    elif FORMAT == "COCO":
        images_coco = []
        labels_coco = []
        categories_coco = [
            {"id": 0, "name": "rummikub-tiles-dataset", "supercategory": None}
        ] + [
            {"id": v+1, "name": k, "supercategory": "rummikub-tiles-dataset"} \
                for k, v in name2class.items()
        ]
        classes_coco = [a["name"] for a in categories_coco]
        
        # Create dirictories
        os.makedirs(f"{PATH}/{PART}", exist_ok=True)
    
    for sample_id in tqdm(range(N_SAMPLES)):
        img, labels = generateSample(sample_id, len(labels_coco))
        if FORMAT == "Yolo":
            cv2.imwrite(f"{PATH}/{PART}/images/{sample_id:06d}.jpg", img)
            with open(f"{PATH}/{PART}/labels/{sample_id:06d}.txt", "w") as f:
                f.write('\n'.join(' '.join(map(str, tile)) for tile in labels))
        elif FORMAT == "COCO":
            cv2.imwrite(f"{PATH}/{PART}/{sample_id:06d}.jpg", img)
            labels_coco.extend(labels)
            images_coco.append({
                "id": sample_id,
                "width": int(img.shape[1]),
                "height": int(img.shape[0]),
                "file_name": f"{sample_id:06d}.jpg"
            })

    if FORMAT == "COCO":
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