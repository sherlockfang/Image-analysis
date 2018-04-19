from pro1_a import *


# sqrt(scaling factors)
scalfact = [1.2 ^ power/2 for power in range(9)]

scalImg = []
for scaleFactor in scalfact:
	scalImg.append(cv2.resize(img, fx=scaleFactor, fy=scaleFactor))