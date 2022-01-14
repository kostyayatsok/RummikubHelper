import cv2
import numpy as np

def show_result(winname, img, wait_time=10):
    scale = 0.1
    disp_img = cv2.resize(img, None, fx=scale, fy=scale)
    cv2.imshow(winname, disp_img)
    cv2.waitKey(wait_time)

def cropTiles(img,
              values = [4, 3, 2, 1],
              colors = ["red", "blue", "black", "orange"]):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # define range of green color in HSV
    lower_green = np.array([60, 30, 15])
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
        one_tile = img[y:y+h, x:x+w]
        
        val = values[(len(values)+1)*y//img.shape[0]]
        col = colors[(len(colors)+1)*x//img.shape[1]]
        path2save = f'images/tiles/{val}_{col}.png'
        cv2.imwrite(path2save, one_tile)
        
        print(path2save, "saved!")

cropTiles(cv2.imread("images/raw/1.jpg"), [4, 3, 2, 1])
cropTiles(cv2.imread("images/raw/2.jpg"), [8, 7, 6, 5])
cropTiles(cv2.imread("images/raw/3.jpg"), [13, 12, 11, 10, 9])
