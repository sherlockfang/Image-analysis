import numpy as np
from sklearn.cluster import KMeans
data = np.random.rand(5000, 1) #生成一个随机数据，样本大小为100, 特征数为3


def hikmeans(data, b, depth):
    list = []#建一个个数等于各层所有nodes总和的空列表
    n = depth - 1
    for i in range(depth):
        for k in range(b ** (depth - n)):
            list.append([])
        n = n - 1
    estimator = KMeans(n_clusters=b)#对原始数据第一次kmeans
    estimator.fit(data)
    label_pred = estimator.labels_

    for j in range(b):#得到第一层各nodes
        for i in range(len(data)):
            if label_pred[i] == j:
                list[j].append(data[i])
    allnodes = hhikmeans(list, b, depth, 1)
    return allnodes


def hhikmeans(data1, b, depth, count):#对第一层nodes进行kmeans，逐层递归
    for n in range(depth-1):
        m = 0
        for l in range(n):
            m = b * m + b
        for i in range(m, m+b**count):
            estimator = KMeans(n_clusters=b)
            estimator.fit(np.array(data1[i]))
            label_pred = estimator.labels_
            for j in range(b):
                for k in range(len(data1[i])):
                    if label_pred[k] == j:
                        data1[b*(i+1)+j].append(data1[i][k])
        count = count + 1
    return data1


allnode = hikmeans(data, 4, 5)