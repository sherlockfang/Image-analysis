# This file mainly implement the construction of a vocabulary tree with K-means clustering as cluster method

import pickle
import numpy
import pdb
from sklearn.cluster import KMeans
from pro2_1 import point


class rankObject(object):
	def __init__(self, id=None, score=None):
		self.id = id
		self.score = score


class queryDescriptorClassify(object):
	def __init__(self, descriptor=None, center=None, distance=None, centerID=None):
		self.descriptor = descriptor
		self.center = center
		self.distance = distance
		self.centerID = centerID

# the problem is each node stores the informatin belonging to its child. The centre info
# of each node should be the center of itself. c1->5 are like pointer to their child.
class treeNode(object):
	def __init__(self, center=None, data=None, c1=None, c2=None, c3=None, c4=None, c5=None):
		'''

		# :param data: it should be 4/5 centroids, it does not have to store all the data
		:param center: the coordinate of the cluster( a descriptor vector). No center for root node
		c1 -> 5: pointer to children
		'''
		self.data = data
		self.center = center
		self.c1 = c1
		self.c2 = c2
		self.c3 = c3
		self.c4 = c4
		self.c5 = c5


class Tree(object):
	'''
	a four-branch tree class
	'''


	def __init__(self, root=None, nodes=[]):
		'''
		This tree has 4 branches at each node
		:param root: the root node of this tree
		:param nodes: the nodes of the tree
		'''
		self.root = root
		self.nodes = nodes


	def addNode(self, data, k, depth, currentDepth=0):
		'''
		Add nodes at next layer
		:param data: data to be clustered
		:param k: the number of branches
		:param depth: the depth of the tree
		:param currentDepth: the round of current iteration

		:return:
		'''
		# the tree has not been completely constructed, so continue to add node
		if currentDepth < depth:


			# node = treeNode(data=data)

			if self.root is None:   # storing data is useless
				self.root = treeNode(center=1)  # just to identify root. Anything except None is fine.
				self.nodes.append(self.root)
				Tree(root=self.root, nodes=self.nodes).addNode(data=data, k=k, depth=depth)

			else:
				parent = self.nodes[0]

				child = [parent.c1, parent.c2, parent.c3, parent.c4, parent.c5]
				centroid, cluster, _ = kmeans(data, k)
				# If the number of cluster < K, stop clustering.
				try:
					for i in range(k):
						child[i] = treeNode(center=centroid[i], data=cluster[i])
						self.nodes.append(child[i])
					self.nodes.pop(0)   # stack
					# all elements in nodes are leaf nodes
					if len(self.nodes) == k**(currentDepth+1):
						currentDepth = currentDepth + 1
					currentTree = Tree(root=self.root, nodes=self.nodes)
					currentTree.addNode(data=self.nodes[0].data, k=k, depth=depth, currentDepth=currentDepth)
				except ValueError:
					print('The node cannot be divided')
		else:
			print('Construction complete')

		return self.nodes


def kmeans(data, k):
	'''

	:param data: the data to be classified into k clusters, in the form of class(descriptor, index)
	:param k: the number of clusters
	:return: a list of centroids, and a list of clusters with corresponding data
	'''

	# extract the descriptor vector from the point class
	desVector = numpy.ndarray(shape=(len(data), 128))
	for i in range(len(data)):
		desVector[i] = numpy.array(data[i].descriptor)
	# pdb.set_trace()
	classifier = KMeans(n_clusters=k).fit(desVector)  # the input of fit should be descriptors!
	centroids = classifier.cluster_centers_
	cluster = classifier.labels_

	# Use the index obtained by clustering desVector. The index in the pointList is the same with in the desVector
	childrenCluster = []
	for j in range(k):
		# singleCluster = numpy.ndarray(shape=(1, 128))
		singleCluster = []
		# j = True # substitute singleCluster[0]
		for i in range(len(data)):
			if cluster[i] == j:
				# if j:
				# 	singleCluster = data[i]
				# 	j = False
				# else:
				singleCluster.append(data[i])
		childrenCluster.append(singleCluster)

	return centroids, childrenCluster, classifier


def decribePoint(data):
	'''

	:param data: input should be :point(class)
	:return: a list of descriptor vector
	'''
	desVector = numpy.ndarray(shape=(len(data.descriptor), 128))
	for i in range(len(data.descriptor)):
		desVector[i] = numpy.array(data.descriptor[i])
	return desVector


def objScoreperQuery(leafNode=None, idf=None, visualWord_object=None, queryDescriptorNumber=None, sumQuery=None, sumIDF=None):
	'''
	:param leafNode: the leaf nodes of the vocabulary tree, that is visual word
	:param idf: inversed document frequency
	:param visualWord_object: a list of objects in all visual word[[visual word1[objects]],[vw2[obj]]
	:param queryDescriptorNumber: list(query descriptor number in each visual word)
	:param sumQuery:list(the sum of query descriptors in visual word)
	:param sumIDF: the IDF for each visual word
	:return: a list of score for all objects in single query
	'''
	objScore = [] # 50 objects' scores
	for obj in range(50):
		score = 0
		for vw in range(len(leafNode)):
			# pdb.set_trace()
			# score = score + (idf[vw]**2 * len(visualWord_object[vw][obj]) * queryDescriptorNumber[vw]) / (sumIDF*sumQuery)
			score = score + (idf[vw]**2 * len(visualWord_object[vw][obj]) * queryDescriptorNumber[vw])/sumIDF
			# score = score + (idf[vw] * queryDescriptorNumber[vw])

		objScore.append(score)
	return objScore


# TODO: Might be the problem
def recallEvaluation(rankObject, top, queryID):
	'''

	:param rankObject: a list query scores in rankObject class (objectID; score of this object)
	:param top: return No 1. -> No top
	:param queryID:
	:return: if match return 1, otherwise
	'''
	rankList = sorted(rankObject, key=lambda rankObject:rankObject.score, reverse=True)
	if top < len(rankList):
		topList = rankList[:top]
		recall = 0
		for match in topList:
			if match.id == queryID:  # queryID ? TODO Mistake
				recall = 1
		return recall


def queryCluster(queryDescriptors, centers):
	'''

	:param queryDescriptors: a list of descriptors from single query
	:param centers: list(center of leaf nodes), the index is the cluster number
	:return: a class
	'''
	queryDesClass = []
	for queryDes in decribePoint(queryDescriptors):
		destoAllcenter = []
		for i in range(len(centers)):
			destoAllcenter.append(queryDescriptorClassify(descriptor=queryDes, center=centers[i],
			                                            distance=numpy.linalg.norm(queryDes - centers[i]), centerID=i))

		queryDesClass.append(sorted(destoAllcenter,
		                            key=lambda queryDescriptorClassify: queryDescriptorClassify.distance)[0])
	# 	the class name before and after : must be the same

	return queryDesClass


def main():

# --------------------------------parameter setting------------------------------------------

	depthBank = [3, 5, 7, 2]
	branchBank = [4, 5, 2]
	depth = depthBank[1]
	branch = branchBank[0]
	percemt = 1
# -------------------------------- Vocabulary tree construction ----------------------------------

	# need to import class defined in other files
	tree = Tree()
	txtPath = '/Users/fangyue/PycharmProjects/Project/Data/descriptorVector'
	with open(txtPath, 'rb') as data:
		pointList = pickle.load(data)

		# The function below is the hi_keams function in the task
		leafNode = tree.addNode(data=pointList, k=branch, depth=depth)

	# read query data
	queryPointPath = '/Users/fangyue/PycharmProjects/Project/Data/queryVector'
	with open(queryPointPath, 'rb') as data:
		queryPoint = pickle.load(data)
		queryPoint = sorted(queryPoint, key=lambda point:point.index)
# -------------------------------- Query retrieval --------------------------------

	# build a classifier based on the leaf nodes' center for query
	# Actually, what I do here is recluster all the leaf nodes for convenience.
	# The centers are different from the leafNode center.
	leafCenters = []
	for node in leafNode:
		leafCenters.append(numpy.array(node.center))

	# _, _, leafClassifier = kmeans(data=pointList, k=branch**depth)
	# leafClassifier.cluster_centers_ = numpy.array(leafCenters)

# 	calculate the number of descriptors from the same object in a visual word (leaf Node)

	# extract the objects ID
	objID = numpy.load('/Users/fangyue/PycharmProjects/Project/Data/databaseID.npy')

	# ------------------------------ Factors in score computation -------------------------
# vwObj =
	# 	[
	# 	 [visual word 1],
	# 	 [visual word 2],
	# 	 ...
	# 	]

# objinVisualWord =
# [
	# [points from obj1],
	# [points from obj2,
	# ...
# ]
	vwObj = [] # store data in all visual word
	idf = []
	for visualWord in leafNode:
		objinVisualWord = []
		for i in objID:  # object ID
			sameObject = []
			for points in visualWord.data:
				if points.index == i: # str
					sameObject.append(points)  # point from the same object
			objinVisualWord.append(sameObject) # obj 01->50
			# objectinVisualWord = list(filter(None, objinVisualWord))
		vwObj.append(objinVisualWord)  # -> C
		idf.append(numpy.log2(50 / len(list(filter(None, objinVisualWord))))) # number of occurence


# 	array([array([]),array([])])
# ------------------------------------ Query retrieval -----------------------TODO
# 	input a query per iteration
	allQueryScore = []
	rankObj = []
	for singleQueryAll in queryPoint:
		singleQueryDes = singleQueryAll.descriptor[:int(percemt * len(singleQueryAll.descriptor))]
		singleQuery = point(descriptor=singleQueryDes, index=singleQueryAll.index)
		# singleQuery = queryPoint[0]
		# predict the which leaf node the query descriptor should belong
		# queryClusterIndex = leafClassifier.predict(decribePoint(singleQuery))
		queryResult = queryCluster(singleQuery, leafCenters)
		queryClusterIndex = [queryDes.centerID for queryDes in queryResult]
		sumQuery = len(numpy.unique(queryClusterIndex)) # sigma_QuertImage

		# pick out queries in the same leaf node(visual word) in a list
		sameClusterQuery = []
		for j in range(len(leafNode)): # every visual word
			currentClusterQuery = []
			for queryDes in queryResult:
				if queryDes.centerID == j:
					currentClusterQuery.append(queryDes)    # the index of query descriptors in the same visual word
			sameClusterQuery.append(currentClusterQuery) # only index of the query

		# for single visual word (vw)
		queryDesNum = []  # -> q: the number of descriptors visiting this visual word
		for visualWord in sameClusterQuery:
			queryDesNum.append(len(visualWord))

# TODO database ID
	# ----------------------- Score computation -----------------------------
		sumIDF = numpy.sum(idf)
		# single query obj 01 -> 50
		singleQueryAllObjectScore = objScoreperQuery(leafNode=leafNode ,idf=idf, visualWord_object=vwObj,
		                                      queryDescriptorNumber=queryDesNum, sumQuery=sumQuery, sumIDF=sumIDF)

		objScore = []
		for i in range(len(singleQueryAllObjectScore)):
			objScore.append(rankObject(id=str(i+1).zfill(2), score=singleQueryAllObjectScore[i])) # transform scores of objects each query
		# into rankObject class

		allQueryScore.append(singleQueryAllObjectScore)
		rankObj.append(recallEvaluation(objScore,top=5, queryID=singleQuery.index))
	# a list of recall value(+:1, -:0)

# ------------------------- Recall rate computation ----------------------------------
	recallRate = numpy.sum(rankObj) / 50
	print(recallRate)
if __name__ == '__main__':
	# from pro2_1 import point
	main()
