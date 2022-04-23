if __name__ == "__main__":
    import sys
    sys.path.append(".")

import glob
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as T
from utils.tools import get_label_encoders
# from utils.plot import plot
from torchvision.io import read_image
from torch.utils.data import Dataset
from typing import Tuple
import cv2
import numpy as np
from scipy import ndimage

class SyntheticConfig:
    background_path: str="images/backgrounds"
    tiles_path: str="images/tiles_aligned"

    dataset_size: int=10000

    scale: Tuple=(0.06, 0.33)
    image_size: int=640
    tile_transforms = nn.Sequential(
        T.RandomRotation(degrees=360, expand=True),
        # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
    )
    total_transforms = nn.Sequential(
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
        T.GaussianBlur(kernel_size=7, sigma=(0.1, 1.5)),
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
        
        self.to_id, self.to_name = get_label_encoders()

    def __len__(self):
        return self.config.dataset_size
    
    def stack_tiles(self, n, resize_fn):
        tiles = []
        bboxes = []
        labels = []

        shift = 0
        for i in range(n):
            type_idx = torch.randint(0, self.n_types, (1,))[0]
            tile_idx = torch.randint(0, len(self.tiles[type_idx]), (1,))[0]
            tile_path = self.tiles[type_idx][tile_idx]
            
            value, color = tile_path.split('/')[-2].split('-')
            labels.append(self.to_id[value+'-'+color])
            
            tile = read_image(tile_path) / 255.
            tile = tile[:,:,int(tile.size(-1)*0.05):-int(tile.size(-1)*0.05)]
            tile = tile[:,int(tile.size(-2)*0.05):-int(tile.size(-2)*0.05),:]
            tile = resize_fn(tile)

            if i > 0 and tile.size(1) != tiles[-1].size(1):
                tile = torch.nn.functional.pad(tile, (0, 0, 0, tiles[-1].size(1) - tile.size(1)))
            
            tiles.append(tile)

            c, h, w = tile.shape
            corners = np.array([shift, 0, w, h])
            bboxes.append(corners)

            shift += w
        return torch.cat(tiles, dim=-1), labels, bboxes

    def rotate_row(self, row, bboxes, max_angle=180):
        angle = (2*torch.rand((1,))[0].item()-1)*max_angle
        origin = (row.size(-1)//2, row.size(-2)//2)
        # origin = (0, 0)
        old_shape = row.shape
        rotated_row = T.functional.rotate(row, angle,expand=True)
        new_shape = rotated_row.shape

        rotated_bboxes = []
        for x0, y0, w, h in bboxes:
            corners = [
                [x0, y0],
                [x0+w, y0],
                [x0, y0+h],
                [x0+w, y0+h],
            ]
            x_min, x_max, y_min, y_max = 1e9, -1e9, 1e9, -1e9
            for corner in corners:
                new_corner = self.rotatePoint(
                    (corner[0], old_shape[1]-corner[1]),
                    origin,
                    angle
                )
                # print((corner[0], old_shape[1]-corner[1]), new_corner)
                x_min = min(new_corner[0], x_min)
                x_max = max(new_corner[0], x_max)
                y_min = min(new_corner[1], y_min)
                y_max = max(new_corner[1], y_max)
            rotated_bboxes.append([x_min, y_max, x_max-x_min, y_max-y_min])
        rotated_bboxes = torch.tensor(rotated_bboxes)
        # print(rotated_bboxes)
        rotated_bboxes[:,0] += (new_shape[-1] - old_shape[-1])//2
        rotated_bboxes[:,1] += (new_shape[-2] - old_shape[-2])//2
        rotated_bboxes[:,1] = new_shape[-2] - rotated_bboxes[:,1]
        rotated_bboxes = torch.maximum(torch.zeros_like(rotated_bboxes), rotated_bboxes)
        return rotated_row, rotated_bboxes

    def rotatePoint(self, p, origin=(0, 0), degrees=0):
        angle = np.deg2rad(degrees)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
        o = np.atleast_2d(origin)
        p = np.atleast_2d(p)
        return np.squeeze((R @ (p.T-o.T) + o.T).T)


    def put_one_row(self, background, tile_size, y0,):
        x0 = 10
        y0 = int(y0)
        background_size = min(background.size(1), background.size(2))
        resize_fn = T.Resize(tile_size, max_size=tile_size+1)

        row, labels, bboxes = self.stack_tiles(background_size//tile_size, resize_fn)
        # print(bboxes)
        # plt.imshow(row.numpy().transpose((1, 2, 0)))
        # plt.show()
        row, bboxes = self.rotate_row(row, bboxes, max_angle=10)
        # print(bboxes)
        # plt.imshow(row.numpy().transpose((1, 2, 0)))
        # plt.show()
        w = min(row.shape[-1], background_size)
        h = min(row.shape[-2], background_size)

        row = row[:,:h,:w]
        mask = (row.sum(0) > 0.)
        if y0+h > background_size:
            return background, [], []
        background[:, y0:y0+h, x0:x0+w][:, mask] = row[:, mask]
        for bbox in bboxes:
            bbox[0] += x0
            bbox[1] += y0
        
        return background, labels, bboxes 


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
        
        labels = []
        bboxes = []
        y0 = 0
        while y0 + 1.5*tile_size < background_size:
            background, label, bbox = self.put_one_row(background, tile_size, y0)
            labels.extend(label)
            bboxes.extend(bbox)
            y0 += 1.5*tile_size

        background, bboxes = self.rotate_row(background, bboxes, max_angle=180)
        background_size = min(background.size(1), background.size(2))

        annotation = [{
            "label": label,
            "x0"   : bbox[0]/background_size,
            "y0"   : bbox[1]/background_size,
            "w"    : bbox[2]/background_size,
            "h"    : bbox[3]/background_size
        } for label, bbox in zip(labels, bboxes)]


        final_image = self.config.total_transforms(background)
        return final_image, annotation


if __name__ == "__main__":
    dataset = SyntheticDataset()
    id2label = {v:k for k, v in dataset.to_id.items()}
    for image, label in dataset:
        plt.imshow(image.numpy().transpose((1, 2, 0)))
        plt.show()
        # plot(image.numpy().transpose((1, 2, 0)), label.numpy(), id2label)