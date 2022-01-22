import cv2
import glob
from generate import name2class
import numpy as np
import os
from toSquare import toSquare
import yaml
import warnings
warnings.filterwarnings('ignore')

IMG_SIZE = 640
# SAVE_PATH = "images/generated/v4/test/"
SAVE_PATH = "images/test/test/"

labels_path = os.path.join(SAVE_PATH, "labels")
images_path = os.path.join(SAVE_PATH, "images")
os.makedirs(labels_path, exist_ok=True)
os.makedirs(images_path, exist_ok=True)

with open("images/test/annotated/data.yaml") as f:
    info = yaml.load(f)

for i, img_path in enumerate(glob.glob("images/test/annotated/train/images/*.jpg")):
    img = cv2.imread(img_path)
    
    txt_path = img_path.split('/')
    txt_path[-2] = "labels"
    txt_path = '/'.join(txt_path)[:-3] + "txt"
    labels = np.loadtxt(txt_path, delimiter=' ')
    bH, bW, _ = img.shape
    new_labels = []
    
    for idx, x0, y0, w, h in labels:
        x0 = x0 * bW / max(bW, bH)
        w  = w  * bW / max(bW, bH)
        y0 = y0 * bH / max(bW, bH)
        h  = h  * bH / max(bW, bH)

        img = toSquare(img)

        name = info["names"][int(idx)]
        val, col = name.split('_')

        new_labels.append([name2class[col], x0, y0, w, h])
        new_labels.append([name2class[val], x0, y0, w, h])
    
    with open(f"{labels_path}/{i:05d}.txt", "w") as f:
        f.write('\n'.join(' '.join(map(str, tile)) for tile in new_labels))
    img = cv2.imwrite(f"{images_path}/{i:05d}.jpg", img)
        
