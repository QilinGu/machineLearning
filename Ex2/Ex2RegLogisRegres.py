__author__ = 'Pratt'

import pylab as pl
import numpy as np
import scipy as sp
import scipy.optimize as op

x1_lists = []
x2_lists = []
y_lists = []

with open('ex2data2.txt') as f:
    for line in f:
        # inner_list = [elt.strip() for elt in line.split(',')]
        # in alternative, if you need to use the file content as numbers
        inner_list = [float(elt.strip()) for elt in line.split(',')]
        x1_lists.append(inner_list[0])
        x2_lists.append(inner_list[1])
        y_lists.append(inner_list[2])

m = len(x1_lists)
# x = np.array(np.column_stack([x1_lists,x2_lists]))

def mapFeature(x1_lists, x2_lists):
    degree = 6
    x1 = np.array(x1_lists)
    # print np.shape(x1)
    x2 = np.array(x2_lists)
    # print np.shape(x2)
    out = np.ones(len(x1_lists))
    # print np.shape(out)
    for i in range(1, degree+1):
        for j in range(0,i+1):
            out = np.array(np.column_stack([out,(x1**(i-j))*(x2**j)]))
            # print np.shape(out), i, j

    return out

x = mapFeature(x1_lists, x2_lists)
[m, n] = np.shape(x)
y = np.array(y_lists)
y.shape = (m,1)
# print y

theta = np.zeros(n)
theta.shape = (n,1)
initial_theta = theta

def sigmoid(z):
    g = 1.0/(1.0+np.exp(-z))
    return g

def gradcost(theta, x, y):
    lamda = 1
    m = len(y)
    # print m
    n = len(theta)
    theta = theta.reshape((n,1))
    # print n
    # print np.shape(theta)
    # print theta
    thetat = theta
    thetat = np.delete(thetat, np.s_[0], axis=0)
    # print np.shape(thetat)
    # print thetat
    thetat = thetat.reshape((n-1,1))
    # print np.shape(thetat)
    # print theta
    # y = y.reshape((m,1))
    term1 = np.log(sigmoid(np.dot(x, theta)))
    # print term1
    term2 = np.log(1-sigmoid(np.dot(x, theta)))
    # print term2
    term3 = (lamda/2.0*m)*(np.dot(thetat.transpose(), thetat))
    term1 = term1.reshape((m,1))
    term2 = term2.reshape((m,1))
    term = y * term1 + (1 - y) * term2
    # print term
    J = -((np.sum(term))/m) + term3
    sigmoid_x_theta = sigmoid(x.dot(theta))
    grad = ((x.T).dot(sigmoid_x_theta-y))/m
    thetat = np.array(np.row_stack([0, thetat]))
    print thetat
    print thetat.shape
    grad = grad + (lamda/m)*thetat
    print J
    print grad
    return J, grad.flatten()

Result = op.minimize(fun = gradcost, x0 = initial_theta, args = (x, y), method = 'TNC', jac = True)

optimal_theta = Result.x
# print np.shape(optimal_theta)
optimal_theta = optimal_theta.reshape((n,1))
print optimal_theta
print np.shape(optimal_theta)
# print np.shape([1, 45, 85])
# print sigmoid(np.dot(optimal_theta, [1, 45, 85]))

def predict(o_theta, x):
    [m, n] = np.shape(x)
    print m, n
    p = np.zeros(m)
    # p = np.shape(m,1)

    for i in range(0,m):

        # x[i,:] = np.reshape(n,1)
        # arrx = x[i,:].reshape((n, 1))
        if (sigmoid(np.dot(o_theta.transpose(),x[i,:]))>0.5): p[i]=1
        else: p[i] = 0

    return p


p = predict(optimal_theta, x)
c = 0.0
for l in range(0,m):
    if (p[l]==y[l]): c = c + 1.0

print 100.0*(c/m)
