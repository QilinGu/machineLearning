__author__ = 'Pratt'

import scipy.io as spio
import numpy as np
import pylab as pl
import scipy.optimize as op
import sklearn.multiclass as sklmc
import sklearn.svm as svm
import sklearn.metrics.pairwise as met
import matplotlib.pyplot as plt

matdata1 = spio.loadmat('ex6data1.mat')

datax = matdata1.get('X')
datay = matdata1.get('y')
X = np.array(datax)
[m, n] = np.shape(X)
print m,n
y = np.array(datay)

clf = svm.LinearSVC(C=1)
clf.fit(X, y)
print clf

def visualizelinearboundary(X,y,model):
    w = model.coef_
    print np.shape(w)
    b = model.intercept_
    print w[0,0], w[0,1], b
    print np.min(X, axis=0)[0]
    print np.max(X, axis=0)[0]

    xp = np.linspace(np.min(X,axis=0)[0], np.max(X, axis=0)[0], 100)
    print xp
    yp = - (w[0,0]*xp + b)/w[0,1]

    pl.scatter(X[:,0],X[:,1],y==1, c='r', marker='|', linewidths=5)
    pl.scatter(X[:,0],X[:,1],y==0, c='g', marker='_', linewidths=5)
    pl.plot(xp, yp)
    pl.show()

visualizelinearboundary(X,y,clf)

def gaussiankernel(x1,x2):

    sigma = 0.1
    sim = np.exp(-(1.0/(2.0*(sigma**2.0)))*(met.euclidean_distances(x1,x2))**2)
    return sim

x1 = [1, 2, 1]
x1 = np.array(x1)
x2 = [0, 4, -1]
x2 = np.array(x2)
# sigma = 2
sim = gaussiankernel(x1, x2)
print sim

matdata2 = spio.loadmat('ex6data2.mat')

datax = matdata2.get('X')
datay = matdata2.get('y')
X = np.array(datax)
[m, n] = np.shape(X)
print m,n
y = np.array(datay)
print np.shape(y.ravel())
print y.ravel()

clf = svm.SVC(C=1, kernel=gaussiankernel)
clf.fit(X, y.ravel())
print clf

h = .02  # step size in the mesh


# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
# print x_min, x_max
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
# print y_min, y_max
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
# print xx, yy
# xx = np.array(xx)
# yy = np.array(yy)

print np.shape(xx.ravel()), np.shape(yy.ravel())
# xx = xx.reshape(100,1)
# yy = yy.reshape(100,1)
print np.shape(np.c_[xx.ravel(), yy.ravel()])
print np.c_[xx.ravel(), yy.ravel()]
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
print Z, np.shape(Z)
"""
xx = xx.ravel()
yy = yy.ravel()
# xx = xx.reshape(10000,1)
# yy = yy.reshape(10000,1)

print np.shape(xx), np.shape(yy)
xx = np.array(xx)
yy = np.array(yy)
Z = clf.predict(np.c_([xx, yy]))
"""
# Put the result into a color plot
Z = Z.reshape(xx.shape)
print np.shape(Z)
plt.pcolormesh(xx, yy, Z)

# Plot also the training points
pl.scatter(X[:, 0], X[:, 1], y==1, 'r', marker='|', linewidths=5)
pl.scatter(X[:, 0], X[:, 1], y==0, 'g', marker='_', linewidths=5)
pl.axis('tight')
plt.show()

matdata3 = spio.loadmat('ex6data3.mat')

datax = matdata3.get('X')
datay = matdata3.get('y')
X = np.array(datax)
[m, n] = np.shape(X)
print m,n
y = np.array(datay)

clf = svm.SVC(C=1, kernel=gaussiankernel)
clf.fit(X, y.ravel())
print clf

x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
# print x_min, x_max
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
# print y_min, y_max
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
# print xx, yy
# xx = np.array(xx)
# yy = np.array(yy)

print np.shape(xx.ravel()), np.shape(yy.ravel())
# xx = xx.reshape(100,1)
# yy = yy.reshape(100,1)
print np.shape(np.c_[xx.ravel(), yy.ravel()])
print np.c_[xx.ravel(), yy.ravel()]
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
print Z, np.shape(Z)

Z = Z.reshape(xx.shape)
print np.shape(Z)
plt.pcolormesh(xx, yy, Z)

# Plot also the training points
pl.scatter(X[:, 0], X[:, 1], y==1, 'r', marker='|', linewidths=5)
pl.scatter(X[:, 0], X[:, 1], y==0, 'g', marker='_', linewidths=5)
pl.axis('tight')
plt.show()