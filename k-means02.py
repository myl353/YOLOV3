# -*- coding: utf-8 -*-
'''
聚类得到各个尺寸的3种建议框
直接调用sklearn的kmeans方法来求
'''

from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

li = []
with open('data1/info.txt',mode='r') as f:
    for line in f.readlines():
        _boxes = np.array([float(x) for x in line.strip().split()[1:]])
        boxes = np.split(_boxes, len(_boxes) // 5)
        for box in boxes:
            cls,cx,cy,w,h = box
            out = w, h
            li.append(out)
li = np.array(li)
# print(a)
kmeans = KMeans(n_clusters=9,init='k-means++')
kmeans.fit(li)
b = kmeans.predict(li)
print(b)
# print(kmeans.cluster_centers_)
w,h = (np.array(kmeans.cluster_centers_)[:,0]).round(2),(np.array(kmeans.cluster_centers_)[:,1]).round(2)
wh=[]
for i in zip(w,h):
    wh.append(i)
    print(i)
#按面积排序
wh.sort(key=lambda x:x[0]*x[1])
print(wh)

plt.scatter(li[:, 0], li[:, 1], c=b)
plt.scatter(w, h, marker='x',c='k')
plt.show()
