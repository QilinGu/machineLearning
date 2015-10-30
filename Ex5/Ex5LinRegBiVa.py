__author__ = 'Pratt'

import scipy.io as spio
import numpy as np
import scipy.optimize as op
import pylab as pl
from scipy.interpolate import interp1d
from scipy.interpolate import spline

import matplotlib.pyplot as mpy

import sklearn.multiclass as sklmc
import sklearn.svm as sklsvm

matdata = spio.loadmat('ex5data1.mat')

# print matdata

datax = matdata.get('X')
datay = matdata.get('y')
dataxval = matdata.get('Xval')
datayval = matdata.get('yval')
dataxtest = matdata.get('Xtest')
dataytest= matdata.get('ytest')
X = np.array(datax)
Xplot = np.array(datax)
y = np.array(datay)
Xval = np.array(dataxval)
yval = np.array(datayval)
Xtest = np.array(dataxtest)
ytest = np.array(dataytest)

[m, n] = np.shape(X)
X = np.array(np.column_stack([np.ones(m),X]))
# Xval = np.array(np.column_stack([np.ones(len(Xval)),Xval]))
# Xtest = np.array(np.column_stack([np.ones(len(Xtest)),Xtest]))

# print X, y, Xval, yval, Xtest, ytest

theta = np.ones(n+1)
theta.shape = (n+1,1)
initial_theta = np.zeros(n+1)
initial_theta.shape = (n+1,1)

def lrcostgrad(theta, X, y):
    # print X.shape, y.shape, theta.shape
    lamda = 1
    m = len(y)
    n = len(theta)-1
    theta = theta.reshape(n+1,1)
    grad = np.zeros(n+1)
    grad = grad.reshape(n+1,1)
    hypothesis = np.dot(X, theta)
    loss = hypothesis - y
    thetat = theta
    thetat = np.delete(thetat, np.s_[0], axis=0)
    regterm = (lamda/(2.0*m))*np.dot(thetat.transpose(), thetat)
    J = np.sum(loss ** 2.0) / (2.0 * m) + regterm
    thetat = np.array(np.row_stack([0.0, thetat]))
    grad = (np.dot(X.transpose(), loss) / m) + (lamda/m)*thetat
    return J, grad

# print lrcostgrad(X, y, theta, 1)

Result = op.minimize(fun = lrcostgrad, x0 = initial_theta, args = (X, y), method = 'TNC', jac = True)
optimal_theta = Result.x
# print optimal_theta

pl.plot(Xplot, y, 'rx')
pl.plot(Xplot, X.dot(optimal_theta))
pl.show()

def learningcurve(X, y, Xval, yval, lamda):
    m,n = np.shape(X)
    # X = np.array(np.column_stack([np.ones(m),X]))
    initial_theta = np.zeros(n)
    initial_theta = initial_theta.reshape(n,1)

    # print np.shape(X)
    error_train = np.zeros(m)
    error_train = error_train.reshape(m,1)
    error_val   = np.zeros(m)
    error_val   = error_val.reshape(m,1)

    for i in range (0,m):
            # print X[:i+1,:]
            # print y[:i+1]
            theta = op.minimize(fun = lrcostgrad, x0 = initial_theta, args = (X[:i+1,:], y[:i+1]), method = 'TNC', jac = True).x
            [error_train[i], grad] = lrcostgrad(theta, X[:i+1,:], y[:i+1])
            [error_val[i], grad] = lrcostgrad(theta, Xval, yval)

    return error_train, error_val

[error_train, error_val] = learningcurve(X, y, np.array(np.column_stack([np.ones(len(Xval)),Xval])), yval, 0)

pl.plot(error_train)
pl.plot(error_val)
pl.show()

def polyfeatures(X,p):
    [m,n]=np.shape(X)
    X_poly = np.zeros(m*p)
    X_poly = X_poly.reshape(m,p)
    for i in range (0,m):
        for j in range(0,p):
            X_poly[i, j] = (X[i])**(j+1)
    return X_poly

X_poly = polyfeatures(Xplot,8)

def featurenormalize(x):
    # x_norm = x
    [m, p] = np.shape(x)
    mu = np.zeros(p)
    sigma = np.zeros(p)

    mu = np.sum(x, axis=0)/m
    sigma = (np.sum((x-mu)**2/(m-1), axis=0))**0.5

    # print mu, np.shape(mu)
    # print sigma, np.shape(sigma)

    x_norm = (x - mu)/sigma
    return x_norm, mu, sigma


[X_poly, mu, sigma] = featurenormalize(X_poly)
X_poly = np.array(np.column_stack([np.ones(m),X_poly]))

# print X_poly

X_poly_val = polyfeatures(Xval,8)
# print np.shape(Xval)
# print X_poly_val, np.shape(X_poly_val)
X_poly_val = (X_poly_val-mu)/sigma
# print X_poly_val, np.shape(X_poly_val)
X_poly_val = np.array(np.column_stack([np.ones(len(Xval)),X_poly_val]))
# print X_poly_val, np.shape(X_poly_val)


X_poly_test = polyfeatures(Xtest,8)
X_poly_test = (X_poly_test-mu)/sigma
X_poly_test = np.array(np.column_stack([np.ones(len(Xtest)),X_poly_test]))

p = 8
initial_theta_poly = np.zeros(p+1)
initial_theta_poly = initial_theta_poly.reshape(p+1,1)

ftheta = (op.minimize(fun = lrcostgrad, x0 = initial_theta_poly, args = (X_poly, y), method = 'TNC', jac = True)).x


# print np.shape(Xplot), np.shape(np.dot(X_poly,ftheta))
pl.plot(Xplot.flatten(), y, 'rx')
xnew = np.linspace(-50, 50, 101)
l = len(xnew)
xnew = xnew.reshape(l,1)
X_poly_new = polyfeatures(xnew,8)
# print np.shape(X_poly_new)
X_poly_new = (X_poly_new-mu)/sigma
# print np.shape(X_poly_new)
X_poly_new = np.array(np.column_stack([np.ones(l),X_poly_new]))
# print np.shape(X_poly_new)
# f = interp1d(xnew, np.dot(X_poly,ftheta), kind='cubic')
pl.plot(xnew,np.dot(X_poly_new,ftheta))
pl.show()

# y_smooth = spline(X_poly_new,np.dot(X_poly_new, ftheta) , xnew)
# pl.plot(xnew, y_smooth, 'red', linewidth=1)
# pl.show()

[error_train, error_val] = learningcurve(X_poly, y, X_poly_val, yval, 0)

pl.plot(error_train)
pl.plot(error_val)
pl.show()