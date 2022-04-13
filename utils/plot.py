import cv2
import glob
import numpy as np
import sys

def plot(img, label, id2label):
    img = img.copy() 
    for idx, x1, y1, w, h in label:
        x2 = x1 + w
        y2 = y1 + h
        color = np.random.randint(0, 255, (3,)) / 255.
        color = (float(color[0]), float(color[1]), float(color[2]))
        # print(img, (x1, y1), (x2, y2))
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        img = cv2.putText(
            img, id2label[int(idx)], (x1, y1+10), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5, color=color, thickness=2
        )

    cv2.imshow('img', img)
    cv2.waitKey(0)

if __name__ == "__main__":
    folder = sys.argv[1]#"images/generated/v6"

    images = sorted(glob.glob(f"{folder}/images/*.jpg"))
    labels = sorted(glob.glob(f"{folder}/labels/*.txt"))

    for img, label in zip(images, labels):
        print(img)
        print(label)
        img = cv2.imread(img)
        label = np.loadtxt(label, delimiter=' ')
        if len(label.shape) == 1:
            label = label.reshape(1, -1)
        print(label)
        plot(img, label)
