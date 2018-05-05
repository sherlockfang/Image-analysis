import cv2

img1 = cv2.imread('obj1_5.jpg', cv2.IMREAD_COLOR)          # queryImage
img2 = cv2.imread('obj1_t1.jpg', cv2.IMREAD_COLOR)  # trainImage
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


descriptor = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.17, edgeThreshold=8)
kps1, features1 = descriptor.detectAndCompute(gray1, None)
kps2, features2 = descriptor.detectAndCompute(gray2, None)
bf = cv2.BFMatcher()
matches = bf.radiusMatch(features1, features2, 150)

img3 = cv2.drawMatchesKnn(img1, kps1, img2, kps2, matches, None, flags=2)
cv2.imshow('match', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('sift_fix_opt_150.png', img3)
