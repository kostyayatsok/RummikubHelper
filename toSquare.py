import cv2
import numpy as np
import glob
import tqdm

def toSquare(img, size=640):
    h, w, c = img.shape
    img = np.pad(img, ((0, max(0, w - h)), (0, max(0, h - w)), (0, 0)))
    img = cv2.resize(img, (size, size))
    return img

if __name__ == "__main__":
    for img_path in tqdm.tqdm(glob.glob("images/test/raw/*.jpg")):
        img = cv2.imread(img_path)
        img = toSquare(img, 640)
        cv2.imwrite(f"images/test/processed/{img_path.split('/')[-1]}", img)