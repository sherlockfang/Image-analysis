import cv2

img = cv2.imread("obj1_5.jpg")

surf = cv2.xfeatures2d.SURF_create(5000)
kp, des = surf.detectAndCompute(img, None)
img = cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('surf_kp', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('surf_kp.png', img)