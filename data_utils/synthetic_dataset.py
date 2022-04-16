import glob
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as T
from utils.plot import plot
from torchvision.io import read_image
from torch.utils.data import Dataset
from typing import Tuple
import cv2
import numpy as np

class SyntheticConfig:
    background_path: str="images/backgrounds"
    tiles_path: str="images/tiles"

    dataset_size: int=10000

    scale: Tuple=(0.05, 0.33)
    image_size: int=640
    tile_transforms = nn.Sequential(
        T.RandomRotation(degrees=360, expand=True),
        # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
    )
    total_transforms = nn.Sequential(
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
        T.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
        T.Resize(640)
    )

class SyntheticDataset(Dataset):
    def __init__(
        self,
        config: SyntheticConfig=SyntheticConfig()
    ) -> None:
        super().__init__()
        
        self.config = config

        self.tiles = []
        for tile_path in glob.glob(f"{config.tiles_path}/*/"): 
            self.tiles.append(glob.glob(f"{tile_path}/*.png"))
        self.backgrounds = glob.glob(f"{config.background_path}/*.jpg")
        self.n_types = len(self.tiles)
        self.n_backgrounds = len(self.backgrounds)
        
        to_id = {}
        for i, c in enumerate(["red","blue","black","orange"]):
            for j, v in enumerate([str(i) for i in range(1,14)] + ["j"]):
                to_id[v+'-'+c] = (j+1)*10 + (i+1)
        self.to_id = to_id

        self.to_name = {v:k for k,v in to_id.items()}

    def __len__(self):
        return self.config.dataset_size
    
    def __getitem__(self, idx: int):
        # choose random background
        background_idx = torch.randint(0, self.n_backgrounds, (1,))[0]
        background = read_image(self.backgrounds[background_idx]) / 255.
        if background.size(0) == 1:
            background = torch.cat([background, background, background], dim=0)
        # crop it to square
        background_size = min(background.size(1), background.size(2))
        background = background[:, :background_size, :background_size]
       
        tile_size = torch.rand(1) * (self.config.scale[1] - \
                        self.config.scale[0]) + self.config.scale[0]
        tile_size = int(background_size * tile_size)
        resize_fn = T.Resize(tile_size)
        x0, y0 = 0, 0
        
        annotation = []
        while x0 < background_size and y0 < background_size:
            type_idx = torch.randint(0, self.n_types, (1,))[0]
            tile_idx = torch.randint(0, len(self.tiles[type_idx]), (1,))[0]
            tile_path = self.tiles[type_idx][tile_idx]
            
            value, color = tile_path.split('/')[-2].split('-')
            
            tile = read_image(tile_path) / 255.
            tile = resize_fn(tile)
            tile = self.config.tile_transforms(tile)
            
            mask = (tile.sum(dim=0) > 0.)
            x, y, w, h = cv2.boundingRect(mask.numpy().astype(np.uint8))
            tile = tile[:, y:y+h, x:x+w]
            mask = (tile.sum(dim=0) > 0.)

            _, tile_h, tile_w = tile.shape
            if x0+tile_w >= background_size:
                x0 = 0
                if len(annotation):
                    y0 += tile_size
                else:
                    y0 = min(y0+tile_size, background_size-tile_size)
            if y0 >= background_size:
                break
            x1 = min(x0+tile_w, background_size)
            y1 = min(y0+tile_h, background_size)
            mask = mask[:y1-y0, :x1-x0]
            background[:, y0:y1, x0:x1][:, mask] = tile[:, :y1-y0, :x1-x0][:, mask]
            annotation.append({
                "label": self.to_id[value+'-'+color],
                "x0": x0 / background_size,
                "y0": y0 / background_size,
                "w": tile_w / background_size,
                "h": tile_h / background_size
            })
            x0 = x1
    
        final_image = self.config.total_transforms(background)
        
        return final_image, annotation

if __name__ == "__main__":
    dataset = SyntheticDataset()
    id2label = {v:k for k, v in dataset.to_id.items()}
    for image, label in dataset:
        plot(image.numpy().transpose((1, 2, 0)), label.numpy(), id2label)