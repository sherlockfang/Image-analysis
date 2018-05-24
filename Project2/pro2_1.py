# This file mainly implemented the storage of image features

# from PIL import Image
import glob
import cv2
import re
import pickle
import numpy

from matplotlib import pyplot as plt


class objectInfo(object):
	'data structure for each picture'

	def __init__(self, descriptor, index, name):
		self.descriptor = descriptor
		# self.keypoint = keypoint
		self.index = index
		self.name = name


class point(object):

	def __init__(self, descriptor, index):
		self.descriptor = descriptor
		self.index = index


def SIFTdescriptorExtraction(path=None, constrastThreshold=None, edgeThreshold=None):
	'''

	:param path: the path of the image folder
	:param constrastThreshold: contrast threshold for the SIFT
	:param edgeThreshold: edge threshold for the SIFT
	:return: a list of object information class; a list of (descriptor, index)
	'''

	sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=constrastThreshold, edgeThreshold=edgeThreshold)
	objects = []
	for imgName in glob.glob(path):
		img = cv2.imread(imgName)
		# use regular expression to filter the object index
		# no extra space in regular expression
		idx = re.search(r'\d{1,2}', re.split(r'\\', imgName)[0]).group()
		kp, des = sift.detectAndCompute(img, None)
		objects.append(objectInfo(descriptor=des, index=str(idx).zfill(2), name=imgName))

		# store every descriptor in the form of [descriptor, index]
	points = []
	for i in range(len(objects)):
		for descriptor in objects[i].descriptor:
			points.append(point(descriptor=descriptor, index=objects[i].index))

	return objects, points



def SIFTdescriptorExtractionperImage(path, contrastThreshold, edgeThreshold):
	'''

	:param path: the path of the image folder
	:param constrastThreshold: contrast threshold for the SIFT
	:param edgeThreshold: edge threshold for the SIFT
	:return: a list of object information class image by image; a list of (descriptor, index)
	'''

	sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=contrastThreshold, edgeThreshold=edgeThreshold)
	objects = []
	points = []

	for imgName in glob.glob(path):
		img = cv2.imread(imgName)
		# use regular expression to filter the object index
		# no extra space in regular expression
		idx = re.search(r'\d{1,2}', re.split(r'\\', imgName)[0]).group()
		kp, des = sift.detectAndCompute(img, None)
		objects.append(objectInfo(descriptor=des, index=str(idx).zfill(2), name=imgName))
		points.append(point(descriptor=des, index=str(idx).zfill(2)))

	return objects, points


def storePoint(path=None, openWay='wb', data=None):
	# if no such file, establish a new one
	file = open(path, openWay)          # must store in binary form
	# store array-like data into a file
	pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)


def pickObjectID(objectInformation):
	'''
		Store the ID of objects

	:param objectInformation: objectInfo class
	:return: a unique list of object IDs
	'''
	objID = []
	for obj in objectInformation:
		objID.append(obj.index)

	return numpy.unique(objID)


def main():
	''''''
	'''
	sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.16, edgeThreshold=9)

	# store data in an object-class list
	objects = []
	imgPath = '/Users/fangyue/PycharmProjects/Project/Data/server/*.JPG'
	for imgName in glob.glob(imgPath):
		img = cv2.imread(imgName)

		# use regular expression to filter the object index
		# no extra space in regular expression
		idx = re.search(r'\d{1,2}', re.split(r'\\', imgName)[1]).group()
		kp, des = sift.detectAndCompute(img, None)
		objects.append(objectInfo(descriptor=des, keypoint=kp, index=str(idx).zfill(2), name=imgName))
	'''

	'''
	# 	sort the lists in the order of index
	objects = sorted(objects, key=lambda visualWord: visualWord.index)
	
	# select the descriptors for the same object into single list, the index of the list = the index of the object - 1
	descriptors = []
	sameObject = []
	index = 1
	for i in range(len(objects)):
		if objects[i].index == index:
			sameObject.append(objects[i].descriptor)
		else:
			# when turn to a new object, store the descriptors for the last object in a list
			descriptors.append([sameObject])
			sameObject = []
			sameObject.append(objects[i].descriptor)
		index = objects[i].index
	'''


	# ------------------------------------  Database Images Info Storage------------------------------
	imgPath = '/Users/fangyue/PycharmProjects/Project/Data/server/*.JPG'
	objectInformation, points = SIFTdescriptorExtraction(path=imgPath, constrastThreshold=0.16, edgeThreshold=9)



	path = '/Users/fangyue/PycharmProjects/Project/Data/databaseID'
	numpy.save(path, pickObjectID(objectInformation))


	databasePath = '/Users/fangyue/PycharmProjects/Project/Data/descriptorVector'   # must with pkl suffix
	storePoint(path=databasePath, data=points)

# -------------------------------------Query Image Info Storage-----------------------------------

	# query image
	queryPath = '/Users/fangyue/PycharmProjects/Project/Data/client/*.JPG'
	objectInformation, queryPoint = SIFTdescriptorExtractionperImage(path=queryPath, contrastThreshold=0.16, edgeThreshold=9)

	queryPointPath = '/Users/fangyue/PycharmProjects/Project/Data/queryVector'
	storePoint(queryPointPath, data=queryPoint)


'''
	txtPath = '/Users/fangyue/PycharmProjects/Project/Data/descriptorVector.pkl'
	with open(txtPath, 'rb') as data:
		pointList = pickle.load(data)
		print(pointList[0])
	'''

if __name__ == '__main__':
	main()