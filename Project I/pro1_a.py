import cv2
from matplotlib import pyplot as plt
import numpy

img = cv2.imread("obj1_5.jpg")

contrastT = 0.05
edgeT = 9

## create a SIFT(class)
sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=contrastT, edgeThreshold=edgeT)
# find keypoints and descriptors
kp, descriptor = sift.detectAndCompute(img, None)
# after using drawKeypoints, the original image changed
cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# extract the coordinates x0,y0 of keypoints
##class keypoint:
##
##	def __init__(self, x, y):
##		self.x = x
##		self.y = y
oriKp = numpy.ndarray((len(kp),2))
for i in range(len(kp)):
	oriKp[i] = (kp[i].pt[0], kp[i].pt[1])


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Image with keypoints')
# plt.show()

##cv2.waitKey(0)
##cv2.destroyAllWindows()
##cv2.imwrite('sift_kp.png', img)


