import glob
import numpy as np
import cv2
import torchvision.transforms as T
import torch

tiles = np.array(
    [cv2.imread(p) for p in glob.glob('images/tiles/*.png')],
    dtype=object
)
n_tiles = len(tiles)

backgrounds = glob.glob('images/backgrounds/*.jpg')
n_backgrounds = len(backgrounds)

transform_fn = T.RandomAffine(
    degrees=360, scale=(0.05, 0.08)
)

def generateSample(min_tiles: int=1, max_tiles: int=100) -> np.array:
    n = np.random.randint(min_tiles, max_tiles)
    idxs = np.random.randint(0, n_tiles, n)
    
    result = cv2.imread(np.random.choice(backgrounds))
    H, W = result.shape[0], result.shape[1]
    labels = []
    for idx, tile in zip(idxs, tiles[idxs]):
        # rotate and scale
        tile = transform_fn(torch.from_numpy(tile.transpose(2, 0, 1)))
        tile = tile.numpy().transpose(1, 2, 0)
        # Find minimum spanning bounding box
        x, y, w, h = cv2.boundingRect(cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY))
        tile = tile[y:y+h, x:x+w, :]
        
        # cv2.imshow('tile', tile)
        # cv2.waitKey(0)
        
        x0 = np.random.randint(0, W-w)
        y0 = np.random.randint(0, H-h)
        mask = (np.sum(tile, axis=-1) != 0)
        result[y0:y0+h, x0:x0+w, :][mask] = tile[mask]
        labels.append([idx, (x0+w/2)/W, (y0+h//2)/H, w/W, h/H])
    # cv2.imshow('collage', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return result, labels #TODO: yolo format

from tqdm import tqdm
for _ in tqdm(range(5000)):
    img, labels = generateSample(1, 2)
    with open(f"images/generated/{_}.txt", "w") as f:
        f.write('\n'.join(' '.join(map(str, tile)) for tile in labels))
    cv2.imwrite(f"images/generated/{_}.jpg", img)
f.close()