import cv2
import glob
import numpy as np
import os
from toSquare import toSquare
def show_result(winname, img, wait_time=10):
    scale = 0.1
    disp_img = cv2.resize(img, None, fx=scale, fy=scale)
    cv2.imshow(winname, disp_img)
    cv2.waitKey(wait_time)

cnt = 0
def cropTiles(raw_img, dir2save):
    global cnt
    hsv = cv2.cvtColor(raw_img, cv2.COLOR_BGR2HSV)
    # define range of green color in HSV
    lower_green = np.array([60, 100, 20])
    upper_green = np.array([90, 255, 255])
    # Threshold the HSV image to extract green color
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.bitwise_not(mask)

    # Smooth edges
    mask = cv2.blur(mask, (15, 15))
    mask = cv2.inRange(mask, 192, 255)
    
    # show_result('tiles', cv2.bitwise_and(img, img, mask=mask), 0)
    labels_map = cv2.connectedComponents(mask)[1]
    labels, counts = np.unique(labels_map, return_counts=True)
    cv2.imwrite(f"{dir2save}/images/{cnt:05d}.jpg", toSquare(raw_img, 416))
    SZ = max(mask.shape)
    with open(f"{dir2save}/labels/{cnt:05d}.txt", "w") as f:
        for l in labels[counts > 1000]:
            if l == 0:
                continue
            one_mask = np.array(labels_map==l, dtype=np.uint8)
            # Find minimum spanning bounding box
            x, y, w, h = cv2.boundingRect(one_mask)
            # filter trash
            if max(w, h) > 3 * min(w, h):
                continue
            f.write(f"0 {(x+w/2)/SZ} {(y+h/2)/SZ} {w/SZ} {h/SZ}\n")
            cnt += 1
    print(f"{cnt:05d} images saved to {dir2save}!", end='\r')



if __name__ == "__main__":
    os.makedirs(f"images/generated/v6/labels", exist_ok=True)
    os.makedirs(f"images/generated/v6/images", exist_ok=True)
    for path in glob.glob("images/raw/mp4/*.mp4"):
        print(f"\n{path} started")
        filename = path.split('/')[-1][:-4]
        
        cap = cv2.VideoCapture(path)
        success, image = cap.read()
        flag = 0
        while success:
            if flag % 4 == 0:
                cropTiles(image, f"images/generated/v6/")
            success, image = cap.read()
            flag += 1
        print(f"\n{path} finished")