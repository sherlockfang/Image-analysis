import cv2
import numpy as np

img1 = cv2.imread('obj1_5.jpg', cv2.IMREAD_COLOR)          # queryImage
img2 = cv2.imread('obj1_t1.jpg', cv2.IMREAD_COLOR)  # trainImage
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.17, edgeThreshold=8)
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=1)
# Need to draw only good matches, so create a mask
matchesMask = [[0, 0] for i in np.arange(len(matches))]
# ratio test as per Lowe's paper
for i, (m) in enumerate(matches):
    matchesMask[i] = [1, 0]
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=0)
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=2)
cv2.imshow('match', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('sift_near neighbour.png', img3)