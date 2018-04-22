import cv2
import numpy as np
import math as mt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def rot_degree(img, degree):
    rows, cols = img.shape
    center = (cols // 2, rows // 2)
    mask = img.copy()
    mask[:, :] = 255
    M = cv2.getRotationMatrix2D(center, degree, 1)
    top_right = np.array((cols - 1, 0)) - np.array(center)
    bottom_right = np.array((cols - 1, rows - 1)) - np.array(center)
    top_right_after_rot = M[0:2, 0:2].dot(top_right)
    bottom_right_after_rot = M[0:2, 0:2].dot(bottom_right)
    new_width = max(int(abs(bottom_right_after_rot[0] * 2) + 0.5), int(abs(top_right_after_rot[0] * 2) + 0.5))
    new_height = max(int(abs(top_right_after_rot[1] * 2) + 0.5), int(abs(bottom_right_after_rot[1] * 2) + 0.5))
    new_center = (new_width // 2, new_height // 2)
    offset_x = (new_width - cols) // 2
    offset_y = (new_height - rows) // 2
    M[0, 2] += offset_x
    M[1, 2] += offset_y
    dst = cv2.warpAffine(img, M, (new_width, new_height))
    mask = cv2.warpAffine(mask, M, (new_width, new_height))
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    return dst, mask, center, new_center

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    center = (cX, cY)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    new_center = (nW // 2, nH // 2)
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    dst = cv2.warpAffine(image, M, (nW, nH))

     # perform the actual rotation and return the image
    return dst, center, new_center


img = cv2.imread('obj1_5.jpg', 0)
sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.16, edgeThreshold=9)
kp, des = sift.detectAndCompute(img, None)
# img1 = cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

a = list(np.arange(0, 375, 15))
num = len(kp)
rpt = []

for x in a:
    dst, center, new_center = rotate_bound(img, x)
    # cen = (np.array(center)).reshape(2, 1
    (cx1, cy1) = center
    # new_cen = (np.array(new_center)).reshape(2, 1)
    (h, w) = dst.shape[:2]
    (cx2, cy2) = (w // 2, h // 2)
    count = 0
    rot_matrix = np.array([[mt.cos(mt.radians(-x)), mt.sin(mt.radians(-x))],
                           [-mt.sin(mt.radians(-x)), mt.cos(mt.radians(-x))]])
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.16, edgeThreshold=9)
    kp1, des1 = sift.detectAndCompute(dst, None)
    act = []
    num1 = len(kp1)
    # img2 = cv2.drawKeypoints(dst, kp1, dst, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    for i in range(0, num):
        matrix_cor = np.array([kp[i].pt[0] - cx1, kp[i].pt[1] - cy1])
        vir = np.dot(rot_matrix, matrix_cor) + np.array([cx2, cy2])
        if count >= num1:
            break
        for j in range(0, num1):
            # act = (np.array(kp1[j].pt)).reshape(2, 1)
            # act = act.tolist()
            act.append(kp1[j].pt)
            if abs(vir[0] - act[j][0]) <= 2 and abs(vir[1] - act[j][1]) <= 2:
                count = count + 1
                break
    ans = count / num
    rpt.insert(x, ans)

plt.plot(a, rpt, color='blue', marker='.')
plt.ylim((0, 1))
plt.xlabel('degree')
plt.ylabel('repeatability')
plt.title('SIFT repeatability with angle')
plt.show()
