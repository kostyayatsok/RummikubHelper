import cv2
import glob
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from tqdm import tqdm

PATH = "images/generated"
N_SAMPLES = 5000
PART = "train"
MAX_ON_ONE = 1
MIN_ON_ONE = 1
IMG_SIZE = 416

os.makedirs(f"{PATH}/{PART}/labels", exist_ok=True)
os.makedirs(f"{PATH}/{PART}/images", exist_ok=True)

# Load tiles and backgrounds images
tiles = np.array(
    [cv2.imread(p) for p in glob.glob('images/tiles/*.png')],
    dtype=object
)
backgrounds = glob.glob('images/backgrounds/*.jpg')

n_tiles = len(tiles)

# Create data.yml
names = [p.split('/')[-1][:-4] for p in glob.glob('images/tiles/*.png')]
with open(f"{PATH}/data.yaml", "w") as data:
    data.write(f"train: {PATH}/train\n")
    data.write(f"val: {PATH}/valid\n")
    data.write(f"nc: {len(names)}\n")
    data.write(f"names: {str(names)}\n")

# Define tiles transformations
tile_aug = nn.Sequential(
    T.RandomAffine(degrees=360, scale=(0.05, 0.08)),
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2, hue=0),
    T.RandomPerspective()
)

# Define composition (whole image) transformations
aug = nn.Sequential(
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2, hue=0),
    T.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))
)


def generateSample(min_tiles: int=1, max_tiles: int=100) -> np.array:
    # Define amount of tiles on sample
    n = np.random.randint(min_tiles, max_tiles+1)
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
    
    labels = []
    for idx, tile in zip(idxs, tiles[idxs]):
        # Convert image to torch tensor
        tile = torch.from_numpy(tile.transpose(2, 0, 1))
        # Transform tile
        tile = tile_aug(tile)
        # Convert torch tensor to numpy array
        tile = tile.numpy().transpose(1, 2, 0)

        # Find minimum spanning bounding box (because of scaling)
        x, y, w, h = cv2.boundingRect(cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY))
        tile = tile[y:y+h, x:x+w, :]
        
        # Define upper-left corner of tile on background
        x0 = np.random.randint(0, bW-w)
        y0 = np.random.randint(0, bH-h)
        # Mask for tile without black background
        mask = (np.sum(tile, axis=-1) != 0)
        # Put tile on background
        result[y0:y0+h, x0:x0+w, :][mask] = tile[mask]

        labels.append(
            [idx, (x0+w/2)/bSize, (y0+h//2)/bSize, w/bSize, h/bSize]
        )
    
    # Convert to torch tensor
    result = torch.from_numpy(result.transpose(2, 0, 1))
    # Transform whole image
    result = aug(result)
    # Convert torch tensor back to numpy array
    result = result.numpy().transpose(1, 2, 0)

    # Resize each sample to one size
    result = cv2.resize(result, (IMG_SIZE, IMG_SIZE))
        
    return result, labels

for _ in tqdm(range(N_SAMPLES)):
    img, labels = generateSample(MIN_ON_ONE, MAX_ON_ONE)
    with open(f"{PATH}/{PART}/labels/{_:06d}.txt", "w") as f:
        f.write('\n'.join(' '.join(map(str, tile)) for tile in labels))
    cv2.imwrite(f"{PATH}/{PART}/images/{_:06d}.jpg", img)