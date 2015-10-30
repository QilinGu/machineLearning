__author__ = 'Pratt'

import pylab as pl
import numpy as np
import scipy.optimize as op

x1_lists = []
x2_lists = []
y_lists = []

with open('ex2data1.txt') as f:
    for line in f:
        # inner_list = [elt.strip() for elt in line.split(',')]
        # in alternative, if you need to use the file content as numbers
        inner_list = [float(elt.strip()) for elt in line.split(',')]
        x1_lists.append(inner_list[0])
        x2_lists.append(inner_list[1])
        y_lists.append(inner_list[2])

# print x1_lists
# print x2_lists
# print y_lists

m = len(x1_lists)
x = np.array(np.column_stack([x1_lists,x2_lists]))
x = np.insert(x,0, values=np.ones(m),axis=1)
x.shape = (m,3)
# print x

y = np.array(y_lists)
y.shape = (m,1)
# print y

theta = np.asarray([0,0,0])
theta.shape = (3,1)
initial_theta = theta

def costGrad(x, y, theta):
    # xTranspose = x.transpose()
    m = len(y)
    J = 0
    grad = np.array(np.zeros(np.shape(theta)))
    # print grad
    for i in range(1, m):
        # print np.log(sigmoid(np.dot(x[i,:],theta)))
        # print np.log(1-sigmoid(np.dot(x[i,:],theta)))
        # J = J + (1/m)*(-y[i]*np.log(sigmoid(x[i,:]*theta))-(1-y[i])*np.log(1-sigmoid(x[i,:]*theta)))
        # grad = grad + (1/m)*(sigmoid(x[i,:]*theta)-y[i])*(x[i,:]).transpose()
        J = J + (1.0/m)*(-y[i]*np.log(sigmoid(np.dot(x[i,:],theta)))-(1-y[i])*np.log(1-sigmoid(np.dot(x[i,:],theta))))
        # print (1.0/m)*(sigmoid(np.dot(x[i,:],theta))-y[i])*xTranspose[:,i]
        deltaGrad = (1.0/m)*(sigmoid(np.dot(x[i,:],theta))-y[i])*np.transpose(x[i,:])
        deltaGrad.shape = (np.shape(theta))
        grad = grad + deltaGrad
    return J, grad

def cost(x, y, theta):
    # xTranspose = x.transpose()
    m = len(y)
    J = 0
    # grad = np.array(np.zeros(np.shape(theta)))
    # print grad
    for i in range(1, m):
        # print np.log(sigmoid(np.dot(x[i,:],theta)))
        # print np.log(1-sigmoid(np.dot(x[i,:],theta)))
        # J = J + (1/m)*(-y[i]*np.log(sigmoid(x[i,:]*theta))-(1-y[i])*np.log(1-sigmoid(x[i,:]*theta)))
        # grad = grad + (1/m)*(sigmoid(x[i,:]*theta)-y[i])*(x[i,:]).transpose()
        J = J + (1.0/m)*(-y[i]*np.log(sigmoid(np.dot(x[i,:],theta)))-(1-y[i])*np.log(1-sigmoid(np.dot(x[i,:],theta))))
        # print (1.0/m)*(sigmoid(np.dot(x[i,:],theta))-y[i])*xTranspose[:,i]
        # deltaGrad = (1.0/m)*(sigmoid(np.dot(x[i,:],theta))-y[i])*np.transpose(x[i,:])
        # deltaGrad.shape = (np.shape(theta))
        # grad = grad + deltaGrad
    return J

def grad(x, y, theta):

    # xTranspose = x.transpose()
    m = len(y)
    # J = 0
    grad = np.array(np.zeros(np.shape(theta)))
    # print grad
    for i in range(1, m):
        # print np.log(sigmoid(np.dot(x[i,:],theta)))
        # print np.log(1-sigmoid(np.dot(x[i,:],theta)))
        # J = J + (1/m)*(-y[i]*np.log(sigmoid(x[i,:]*theta))-(1-y[i])*np.log(1-sigmoid(x[i,:]*theta)))
        # grad = grad + (1/m)*(sigmoid(x[i,:]*theta)-y[i])*(x[i,:]).transpose()
        # J = J + (1.0/m)*(-y[i]*np.log(sigmoid(np.dot(x[i,:],theta)))-(1-y[i])*np.log(1-sigmoid(np.dot(x[i,:],theta))))
        # print (1.0/m)*(sigmoid(np.dot(x[i,:],theta))-y[i])*xTranspose[:,i]
        deltaGrad = (1.0/m)*(sigmoid(np.dot(x[i,:],theta))-y[i])*np.transpose(x[i,:])
        deltaGrad.shape = (np.shape(theta))
        grad = grad + deltaGrad

    return grad

def sigmoid(z):
    # g = np.zeros(np.shape(np.array(z)))
    # print np.shape([1,2])
    # [k, l] = np.shape(g)
    # for i in range(1, k):
        # for j in range(1, l):
            # g[i,j] = 1/(1+np.exp(-z[i,j]))
    g = 1.0/(1.0+np.exp(-z))
    return g

J = cost(x, y, theta)
grad = grad(x, y, theta)
print J
print grad

Result = op.minimize(fun = costGrad, x0 = initial_theta, args = (x, y), method = 'BFGS')
# optimal_theta = Result.x