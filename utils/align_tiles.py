import glob
import os
from traceback import print_tb
from matplotlib import pyplot as plt
import numpy as np
# import cv2 as cv
import cv2
from tqdm import tqdm
from scipy import ndimage

def align1(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray,2,3,0.04)
    #result is dilated for marking the corners, not important
    dst = cv.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.32779*dst.max()]=[0,0,255]
    return img

def align2(image):
    def rotate_image(image, angle):
        # Grab the dimensions of the image and then determine the center
        (h, w) = image.shape[:2]
        (cX, cY) = (w / 2, h / 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # Compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # Adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # Perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))

    def order_points_clockwise(pts):
        # sort the points based on their x-coordinates
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        # now, sort the right-most coordinates according to their
        # y-coordinates so we can grab the top-right and bottom-right
        # points, respectively
        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        # return the coordinates in top-left, top-right,
        # bottom-right, and bottom-left order
        return np.array([tl, tr, br, bl], dtype="int32")

    def perspective_transform(image, corners):
        def order_corner_points(corners):
            # Separate corners into individual points
            # Index 0 - top-right
            #       1 - top-left
            #       2 - bottom-left
            #       3 - bottom-right
            corners = [(corner[0][0], corner[0][1]) for corner in corners]
            top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]
            return (top_l, top_r, bottom_r, bottom_l)

        # Order points in clockwise order
        ordered_corners = order_corner_points(corners)
        top_l, top_r, bottom_r, bottom_l = ordered_corners

        # Determine width of new image which is the max distance between 
        # (bottom right and bottom left) or (top right and top left) x-coordinates
        width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
        width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
        width = max(int(width_A), int(width_B))

        # Determine height of new image which is the max distance between 
        # (top right and bottom right) or (top left and bottom left) y-coordinates
        height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
        height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
        height = max(int(height_A), int(height_B))

        # Construct new points to obtain top-down view of image in 
        # top_r, top_l, bottom_l, bottom_r order
        dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], 
                        [0, height - 1]], dtype = "float32")

        # Convert to Numpy format
        ordered_corners = np.array(ordered_corners, dtype="float32")

        # Find perspective transform matrix
        matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

        # Return the transformed image
        return cv2.warpPerspective(image, matrix, (width, height))

    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 120, 255, 1)

    corners = cv2.goodFeaturesToTrack(canny,4,0.5,50)

    c_list = []
    for corner in corners:
        x,y = corner.ravel()
        c_list.append([int(x), int(y)])
        cv2.circle(image,(x,y),5,(36,255,12),-1)

    corner_points = np.array([c_list[0], c_list[1], c_list[2], c_list[3]])
    ordered_corner_points = order_points_clockwise(corner_points)
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [ordered_corner_points], (255,255,255))

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        if len(approx) == 4:
            transformed = perspective_transform(original, approx)

    result = rotate_image(transformed, -90)

    cv2.imshow('canny', canny)
    cv2.imshow('image', image)
    cv2.imshow('mask', mask)
    cv2.imshow('transformed', transformed)
    cv2.imshow('result', result)
    cv2.waitKey()

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    # if len(result.shape) == 2:
    #     x, y, w, h = cv2.boundingRect(result)
    #     result = result[y:y+h, x:x+w]
    # else:
    #     mask = (result.sum(2) > 0).astype(np.uint8) * 255
    #     # cv2.imshow("mask", mask)
    #     x, y, w, h = cv2.boundingRect(mask)
    #     result = result[y:y+h, x:x+w, :]
    return result
def align3(image):

    original = image.copy()
    image = (image.sum(2) > 0).astype(np.uint8) * 255

    l, r = 0, 90
    for i in range(100):
        m1 = l+(r-l)/3
        m2 = r-(r-l)/3

        img1 = rotate_image(image, m1)
        img2 = rotate_image(image, m2)

        score1 = (img1 == 0).sum()
        score2 = (img2 == 0).sum()

        # cv2.imshow("img1", img1)
        # cv2.imshow("img2", img2)
        # cv2.waitKey(10)

        if score1 < score2:
            r = m2
        else:
            l = m1
    result = rotate_image(original, l)
    if result.shape[0] < result.shape[1]:
        result = result.transpose([1,0,2])
    return result

def align(image):
    original = image.copy()
    image = (image.sum(2) > 0).astype(np.uint8) * 255
    ret,thresh = cv2.threshold(image,127,255,0)
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # img = cv2.drawContours(original,[box],0,(0,0,255),2)
    # plt.imshow(img)
    # plt.show()
    # return original
    result = ndimage.rotate(original, rect[-1])

    mask = (result.sum(2) > 0).astype(np.uint8) * 255
    x, y, w, h = cv2.boundingRect(mask)
    result = result[y:y+h, x:x+w, :]
    
    if result.shape[0] < result.shape[1]:
        result = ndimage.rotate(result, 90)
    return result

if __name__ == "__main__":
    os.makedirs("images/tiles_aligned/", exist_ok=True)
    for tile_dir in tqdm(glob.glob("images/tiles/*")):
        new_tile_dir = "images/tiles_aligned/"+tile_dir.split('/')[-1]
        os.makedirs(new_tile_dir, exist_ok=True)
        for image_path in glob.glob(tile_dir+"/*.png"):
            image = cv2.imread(image_path)
            # image = np.pad(image, ((100, 100), (100, 100), (0, 0)))
            aligned = align(image)
            # cv2.imshow("test.png", aligned)
            # cv2.imwrite("test.png", aligned)
            # cv2.waitKey(100)
            # for i in range(100000000000): pass
            # cv2.destroyAllWindows()
            new_image_path = image_path.split('/')
            new_image_path[1] = "tiles_aligned"
            new_image_path = "/".join(new_image_path)
            cv2.imwrite(new_image_path, aligned)