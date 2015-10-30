__author__ = 'Pratt'

import scipy.io as spio
import numpy as np
import scipy.misc as misc

matdata = spio.loadmat('ex7data2.mat')

datax = matdata.get('X')
X = np.array(datax)
[m, n] = np.shape(X)
# print X
# print m,n

K = 3
initial_centroids = [[3, 3],[6, 2],[8, 5]]
initial_centroids = np.array(initial_centroids)
# initial_centroids = initial_centroids.reshape(np.shape(initial_centroids))
# print initial_centroids, np.shape(initial_centroids)


def findClosestCentroids(X, centroids):
    k,l = np.shape(centroids)
    m, n = np.shape(X)
    idx = np.zeros(m)
    idx = idx.reshape(m,1)

    for i in range (0,m):
        minDist = np.dot((X[i,:]-centroids[1,:]),(X[i,:]-centroids[1,:]).transpose())
        idx[i] = 1
        for j in range (0,k):
            if (np.dot((X[i,:]-centroids[j,:]),(X[i,:]-centroids[j,:]).transpose()) < minDist): minDist = np.dot((X[i,:]-centroids[j,:]),(X[i,:]-centroids[j,:]).transpose());idx[i] = j

    return idx


idx = findClosestCentroids(X, initial_centroids)
# print idx

def computeCentroids(X, idx, k):
    m, n = np.shape(X)
    centroids = np.zeros(k*n)
    centroids = centroids.reshape(k,n)
    for i in range(0,k):
        sum = np.zeros(n)
        sum = sum.reshape(1,n)
        count = 0
        for j in range (0,m):
            if (idx[j]==i): sum += X[j,:]; count+=1
        centroids[i,:]= sum/count

    return centroids

# in_centroids = computeCentroids(X, idx, K)
# print in_centroids

def runkMeans(X,initial_centroids, max_iters ):

    centroids = initial_centroids
    for n in range(0,max_iters):
        idx = findClosestCentroids(X, centroids)
        centroids = computeCentroids(X, idx, 3)
        print centroids

    return centroids, idx


[centroids, idx] = runkMeans(X, initial_centroids, 10)

print centroids



