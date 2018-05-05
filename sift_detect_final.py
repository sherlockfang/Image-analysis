import cv2
import numpy as np
import math as mt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def rot_degree(img, degree):
    rows, cols = img.shape
    center = (cols // 2, rows // 2)
    # mask = img.copy()
    # mask[:, :] = 255
    M = cv2.getRotationMatrix2D(center, degree, 1)
    top_right = np.array((cols - 1, 0)) - np.array(center)
    bottom_right = np.array((cols - 1, rows - 1)) - np.array(center)
    top_right_after_rot = M[0:2, 0:2].dot(top_right)
    bottom_right_after_rot = M[0:2, 0:2].dot(bottom_right)
    new_width = max(int(abs(bottom_right_after_rot[0] * 2) + 0.5), int(abs(top_right_after_rot[0] * 2) + 0.5))
    new_height = max(int(abs(top_right_after_rot[1] * 2) + 0.5), int(abs(bottom_right_after_rot[1] * 2) + 0.5))
    # new_center = (new_width // 2, new_height // 2)
    offset_x = (new_width - cols) // 2
    offset_y = (new_height - rows) // 2
    M[0, 2] += offset_x
    M[1, 2] += offset_y
    dst = cv2.warpAffine(img, M, (new_width, new_height))
    # mask = cv2.warpAffine(mask, M, (new_width, new_height))
    # _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    return dst


gray = cv2.imread('obj1_5.jpg', 0)
descriptor = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.16, edgeThreshold=9)
kps, features = descriptor.detectAndCompute(gray, None)
list = []
(h1, w1) = gray.shape[:2]
(cX1, cY1) = (w1 // 2, h1 // 2)
for k in range(0, 375, 15):
    count = 0
    j = rot_degree(gray, k)
    (h2, w2) = j.shape[:2]
    (cX2, cY2) = (w2 // 2, h2 // 2)
    descriptor = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.16, edgeThreshold=9)
    kps1, features1 = descriptor.detectAndCompute(j, None)
    list1 = []
    for m in range(len(kps1)):
        list1.append(kps1[m].pt)
    for n in range(len(kps)):
        matrix_cor = np.array([kps[n].pt[0]-cX1, kps[n].pt[1]-cY1])
        matrix_rotate = np.array([[mt.cos(mt.radians(k)), mt.sin(mt.radians(k))],
                                  [-mt.sin(mt.radians(k)), mt.cos(mt.radians(k))]])
        cor_new = np.dot(matrix_rotate, matrix_cor)+np.array([cX2, cY2])
        if count >= len(list1):
            break
        for p in range(len(kps1)):
            if abs(cor_new[0]-list1[p][0]) <= 2:
                if abs(cor_new[1]-list1[p][1]) <= 2:
                    count += 1
                    break
    list.append(count/len(kps))
x = range(0, 375, 15)
y = list
plt.plot(x, y, color='blue', marker='.')
plt.ylim((0, 1))
plt.xlabel('degree')
plt.ylabel('repeatability')
plt.title('SIFT repeatability with angle')
plt.show()
