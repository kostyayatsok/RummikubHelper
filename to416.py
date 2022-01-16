import cv2
import numpy as np
import glob
import tqdm


for img_path in tqdm.tqdm(glob.glob("images/test/raw/*.jpg")):
    img = cv2.imread(img_path)
    h, w, c = img.shape
    img = np.pad(img, ((0, max(0, w - h)), (0, max(0, h - w)), (0, 0)))
    img = cv2.resize(img, (416, 416))
    cv2.imwrite(f"images/test/processed/{img_path.split('/')[-1]}", img)