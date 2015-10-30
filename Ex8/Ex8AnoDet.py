__author__ = 'Pratt'

import scipy.io as spio
import numpy as np
import pylab as pl
import numpy.linalg as lalg


matdata = spio.loadmat('ex8data1.mat')
# print matdata

datax = matdata.get('X')
X = np.array(datax)
[m, n] = np.shape(X)
# print X, m, n

dataxval = matdata.get('Xval')
Xval = np.array(dataxval)
[o, p] = np.shape(Xval)
# print o,p

datayval = matdata.get('yval')
yval = np.array(datayval)

pl.plot(X[:,0], X[:,1], 'bo')
# pl.show()
pl.plot(Xval[:,0], Xval[:,1], 'rx')
# pl.show()

def estimateGaussian(X):

    [m, n] = np.shape(X)
    mu = np.zeros(n)
    sigma = np.zeros(n)

    mu = np.sum(X, axis=0)/m
    sigma2 = (np.sum((X-mu)**2/(m-1), axis=0))

    mu = mu.reshape(1,n)
    sigma2 = sigma2.reshape(1,n)
    return mu, sigma2

mu, sigma2 = estimateGaussian(X)
# print mu, sigma2, np.shape(mu), np.shape(sigma2)

def multivariateGaussian(X, mu, sigma2):
    m,n = np.shape(sigma2)
    if ((m==1)or(n==1)): sigmaMat = np.diag(np.matrix(sigma2).A1)

    XsigmaMat = np.multiply(np.dot(X-mu, lalg.pinv(sigmaMat)), X-mu)
    # print XsigmaMat, np.shape(XsigmaMat)
    p = ((2*np.pi)**(-n/2))*((lalg.det(sigmaMat))**(-0.5))*np.exp(-0.5*np.sum(XsigmaMat,axis=1))

    return p

p = multivariateGaussian(X, mu, sigma2)

pval = multivariateGaussian(Xval, mu, sigma2)
# pval = pval.reshape(o,1)

# print pval, np.shape(pval)

def selectThreshold(yval, pval):

    bestEpsilon = 0
    bestF1 = 0
    F1 = 0

    start = np.min(pval, axis=0)
    end = np.max(pval, axis=0)
    # print start, end

    stepsize = (end - start) / 1000
    # print stepsize
    # range = np.arange(np.min(pval, axis=0), np.max(pval, axis=0), stepsize)

    for epsilon in np.arange(np.min(pval, axis=0), np.max(pval, axis=0), stepsize):
            # print predictions, np.shape(predictions)
            predictions = (np.zeros(np.shape(yval)))
            k,l = np.shape(yval)
            # print np.shape(predictions), np.shape(yval), np.shape(pval)
            for i in range(0,k):
                if (pval[i] < epsilon): predictions[i] = 1; notpredictions = 0
                else: predictions[i] = 0; notpredictions = 1

            # print predictions
            tp = np.sum(np.multiply(yval,predictions))
            fp = np.sum(np.multiply(np.invert(yval),predictions))
            fn = np.sum(np.multiply(yval,notpredictions))

            prec = tp/(tp+fp)
            rec = tp/(tp+fn)
            F1 = 2.0*prec*rec/(prec+rec)

            if (F1 > bestF1): bestF1 = F1; bestEpsilon = epsilon

    return bestEpsilon, bestF1

epsilon, F1 = selectThreshold(yval,pval)

print epsilon, np.sum(p<epsilon)

matdata2 = spio.loadmat('ex8data2.mat')

datax = matdata2.get('X')
X = np.array(datax)
[m, n] = np.shape(X)
# print X, m, n

dataxval = matdata2.get('Xval')
Xval = np.array(dataxval)
[q, r] = np.shape(Xval)
# print o,p

datayval = matdata2.get('yval')
yval = np.array(datayval)

mu, sigma2 = estimateGaussian(X)
p = multivariateGaussian(X, mu, sigma2)
pval = multivariateGaussian(Xval, mu, sigma2)

epsilon, F1 = selectThreshold(yval,pval)

print epsilon

outliers = np.sum(p<epsilon)

print outliers