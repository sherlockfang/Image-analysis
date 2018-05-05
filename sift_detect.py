import cv2

img = cv2.imread("obj1_5.jpg")

sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.16, edgeThreshold=9)
kp, des = sift.detectAndCompute(img, None)
img = cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('sift_kp', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite('sift_kp.png', img)