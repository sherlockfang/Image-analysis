import cv2

img1 = cv2.imread('obj1_5.jpg', cv2.IMREAD_COLOR)          # queryImage
img2 = cv2.imread('obj1_t1.jpg', cv2.IMREAD_COLOR)  # trainImage
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


descriptor = cv2.xfeatures2d.SURF_create(6000)
kps1, features1 = descriptor.detectAndCompute(gray1, None)
kps2, features2 = descriptor.detectAndCompute(gray2, None)
bf = cv2.BFMatcher()
matches = bf.knnMatch(features1, features2, k=2)
good = []
for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

img3 = cv2.drawMatches(img1, kps1, img2, kps2, good, None, flags=2)
cv2.imshow('match', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('surf_knn_opt_75.png', img3)
