import cv2
import glob
import numpy as np
import os

def show_result(winname, img, wait_time=10):
    scale = 0.1
    disp_img = cv2.resize(img, None, fx=scale, fy=scale)
    cv2.imshow(winname, disp_img)
    cv2.waitKey(wait_time)

cnt = 0
def cropTiles(img, dir2save):
    global cnt
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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
    img = cv2.bitwise_and(img, img, mask=mask)
    labels_map = cv2.connectedComponents(mask)[1]
    labels, counts = np.unique(labels_map, return_counts=True)
    for l in labels[counts > 1000]:
        if l == 0:
            continue
        one_mask = np.array(labels_map==l, dtype=np.uint8)
        # Find minimum spanning bounding box
        x, y, w, h = cv2.boundingRect(one_mask)
        # filter trash
        if max(w, h) > 8 * min(w, h):
            continue
        one_tile = img[y:y+h, x:x+w]
        
        cnt += 1
        cv2.imwrite(os.path.join(dir2save, f"{cnt:05d}.png"), one_tile)
        # cv2.imshow(dir2save, one_tile)
        # cv2.waitKey(20)
        print(f"{cnt:05d} images saved to {dir2save}!", end='\r')

for path in glob.glob("images/raw/mp4/*.mp4"):
    print(f"\n{path} started")
    cnt = 0
    filename = path.split('/')[-1][:-4]
    os.makedirs(f"images/tiles/{filename}", exist_ok=True)
    
    cap = cv2.VideoCapture(path)
    success, image = cap.read()
    while success:
        cropTiles(image, f"images/tiles/{filename}")
        success, image = cap.read()
    print(f"\n{path} finished")