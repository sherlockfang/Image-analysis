import cv2
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

scalfact = [1.2 ** power for power in range(9)]
gray = cv2.imread('obj1_5.jpg', 0)
descriptor = cv2.xfeatures2d.SURF_create(8000)
kps, features = descriptor.detectAndCompute(gray, None)
list = []

for k in scalfact:
    count = 0
    scal = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    kps1, features1 = descriptor.detectAndCompute(scal, None)
    list1 = []
    for m in range(len(kps1)):
        list1.append(kps1[m].pt)
    for n in range(len(kps)):
        pre = np.array([kps[n].pt[0] * k, kps[n].pt[1] * k])
        if count >= len(list1):
            break
        for p in range(len(kps1)):
            if abs(pre[0]-list1[p][0]) <= 2:
                if abs(pre[1]-list1[p][1]) <= 2:
                    count += 1
                    break
    list.append(count/len(kps))
x = scalfact
y = list
plt.plot(x, y, color='blue', marker='.')
plt.ylim((0, 1))
plt.xlabel('Rescaling Factor')
plt.ylabel('Repeatability')
plt.title('SURF repeatability with Rescaling Factor')
plt.show()