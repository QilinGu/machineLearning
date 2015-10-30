__author__ = 'Pratt'

import scipy.io as spio
import scipy.linalg as lalg
import numpy as np
import pylab as pl


matdata = spio.loadmat('ex7data1.mat')

datax = matdata.get('X')
X = np.array(datax)
[m, n] = np.shape(X)
print X

def featurenormalize(x):
    # x_norm = x
    [m, n] = np.shape(x)
    mu = np.zeros(n)
    sigma = np.zeros(n)

    mu = np.sum(x, axis=0)/m
    print mu
    sigma = (np.sum((x-mu)**2/(m-1), axis=0))**0.5
    print sigma

    # print mu, np.shape(mu)
    # print sigma, np.shape(sigma)

    x_norm = (x - mu)/sigma
    print x_norm
    return x_norm

X_norm = featurenormalize(X)
# print X_norm, mu, sigma

def pca(x):
    [m, n] = np.shape(x)
    covar = (1.0/m)*np.dot(x.transpose(),x)
    print covar
    [U, s, Vh] = lalg.svd(covar)

    return U, s


U, s = pca(X_norm)

print U, s

def projectData(X, U, k):
    m, n = np.shape(X)
    Z = np.zeros(m*k)
    Z = Z.reshape(m, k)

    Z = np.dot(X , U[:,:k])

    return Z

Z = projectData(X_norm, U, 1)
print Z

def recoverData(Z, U, k):
    x, y = np.shape(Z)
    u, v = np.shape(U)
    X_rec = np.zeros(x*u)
    X_rec = X_rec.reshape(x,u)

    X_rec = np.dot(Z,U[:,:k].transpose())

    return X_rec

X_rec = recoverData(Z, U, 1)
print X_rec


facedata = spio.loadmat('ex7faces.mat')

datax = facedata.get('X')
X = np.array(datax)
[m, n] = np.shape(X)
print X, m, n

X_norm = featurenormalize(X)

U, s = pca(X_norm)

Z = projectData(X_norm, U, 100)
print Z, np.shape(Z)

X_rec = recoverData(Z, U, 100)
print X_rec, np.shape(X_rec)

dispX = X[0,:]
dispX = dispX.reshape(np.sqrt(n), np.sqrt(n))

pl.imshow(dispX, cmap='Greys_r')
pl.show()
pl.imshow(X_rec[0,:].reshape(32,32), cmap='Greys_r')
pl.show()
