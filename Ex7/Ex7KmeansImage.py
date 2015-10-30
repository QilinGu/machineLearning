__author__ = 'Pratt'

import numpy as np
import scipy.misc as misc
import pylab as pl
# import matplotlib.cm as CM

image = misc.imread('bird_small.png')

image = np.array(image)
image = image/255.0
m,n,p = np.shape(image)
image = image.reshape(m*n, 3)
print image, np.shape(image)

k = 16
max_iters = 10

def kmeansInitCentroids(X,k):
    mn, p = np.shape(X)
    randidx = np.random.permutation(mn)
    centroids = X[randidx[1:k],:]

    return centroids

initial_centroids = kmeansInitCentroids(image,k)

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

def runkMeans(X,initial_centroids, max_iters ):

    centroids = initial_centroids
    for n in range(0,max_iters):
        idx = findClosestCentroids(X, centroids)
        centroids = computeCentroids(X, idx, 16)
        print centroids

    return centroids, idx

centroids, idx = runkMeans(image, initial_centroids, max_iters)

idx = findClosestCentroids(image, centroids)
print idx, np.shape(idx)

X_recovered = np.zeros(m*n*3)
X_recovered = X_recovered.reshape(m*n, 3)
for i in range(0,len(idx)):
    X_recovered[i,:] = centroids[int(idx[i]),:]

print X_recovered, np.shape(X_recovered)
m,n,p = np.shape(misc.imread('bird_small.png'))

X_recovered = X_recovered.reshape(m,n,p)

# pl.imshow(misc.imread('bird_small.png'))
pl.imshow(X_recovered)

pl.show()



