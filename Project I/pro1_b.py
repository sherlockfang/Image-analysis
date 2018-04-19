from pro1_a import *
import numpy

## (b) Rotation and descriptor match


def square_match(originalKp, rotateKp):

	'''

	:param originalKp: the keypoint before rotation
	:param rotateKp:
	:return: if match, return 1
	'''
	if abs(originalKp[0] - rotateKp[0]) <2 && abs(originalKp[1] - rotateKp[1]) < 2
	return 1


img = cv2.imread("obj1_5.jpg")

rows, cols = img.shape[:2]
# Rotation matrix
# Rotation centre = the centre of the image
# Rotation degreee = 0:15:360
# for d in range(0, 360, 15):
rotDegree = 90
roM = cv2.getRotationMatrix2D((cols/2, rows/2), rotDegree, 1)
roImg = cv2.warpAffine(img,roM,(cols,rows))
rotM = numpy.array([[roM[0][0],roM[0][1]],[roM[1][0],roM[1][1]]])

plt.imshow(cv2.cvtColor(roImg,cv2.COLOR_BGR2RGB))
plt.title('Image after rotation')
# plt.show()


# calculate repeatability

idealKp = numpy.ndarray((len(oriKp),2))
centroid = numpy.array([cols/2,  rows/2])

## compute ideal keypoints
for i in range(len(kp)):
    # idealKp[i] = numpy.dot(rotM, (oriKp[i].transpose() - centroid)) + centroid
    idealKp[i] = numpy.dot((oriKp[i] - centroid), rotM) + centroid
    # idealKp[i] = roM.transpose * (oriKp[i] - centroid.transpose()) + centroid
# got stuck in array mulplication for quite a long time, make sure +/- matrices in the same size

roKp, roDescriptor = sift.detectAndCompute(roImg, None)
# cv2.DescriptorMatcher.radiusMatch(queryDescriptors=roDescriptor, trainDescriptors=descriptor, maxDistance=2)

# Actual Keypoints after rotation
rotKp = numpy.ndarray((len(roKp),2))
for i in range(len(roKp)):
	rotKp[i] = (roKp[i].pt[0], roKp[i].pt[1])

match = 0
for originalKp in oriKp:
	for rotateKp in rotKp:
		match = match + square_match(originalKp, rotateKp)

repeatability = match / len(oriKp)