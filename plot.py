import cv2
import glob
import numpy as np
from generate import classes_names

def plot(img, label):
    H, W, _ = img.shape
    for idx, x, y, w, h in label:
        x *= W
        w *= W
        y *= H
        h *= H

        x1 = int(x - w // 2)
        y1 = int(y - h // 2)
        x2 = int(x + w // 2)
        y2 = int(y + h // 2)

        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 1)
        img = cv2.putText(img, classes_names[int(idx)] if idx <= 13 else "   " + classes_names[int(idx)], (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)

if __name__ == "__main__":
    folder = "images/generated/v4/test"

    images = sorted(glob.glob(f"{folder}/images/*.jpg"))
    labels = sorted(glob.glob(f"{folder}/labels/*.txt"))

    for img, label in zip(images, labels):
        print(img)
        print(label)
        img = cv2.imread(img)
        label = np.loadtxt(label, delimiter=' ')

        plot(img, label)
