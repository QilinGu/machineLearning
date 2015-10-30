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
# print initial_theta

"""
def costGrad(x, y, theta):
    # xTranspose = x.transpose()
    m = len(y)
    J = 0
    grad = np.array(np.zeros(np.shape(theta)))
    # print grad
    for i in range(1, m):

        term1 = np.log(sigmoid(np.dot(x[i,:],theta)))
        term2 = np.log(1-sigmoid(np.dot(x[i,:], theta)))
        J = J + (1.0/m)*(-y[i]*term1 - (1-y[i])*term2)
        # J = J + (1.0/m)*(-y[i]*np.log(sigmoid(np.dot(x[i,:],theta)))-(1-y[i])*np.log(1-sigmoid(np.dot(x[i,:],theta))))
        deltaGrad = (1.0/m)*(sigmoid(np.dot(x[i,:],theta))-y[i])*np.transpose(x[i,:])
        deltaGrad.shape = (np.shape(theta))
        grad = grad + deltaGrad
    return J, grad

def cost(x, y, theta):
    m,n = x.shape
    theta = theta.reshape((n,1))
    y = y.reshape((m,1))
    term1 = np.log(sigmoid(x.dot(theta)))
    term2 = np.log(1-sigmoid(x.dot(theta)))
    term1 = term1.reshape((m,1))
    term2 = term2.reshape((m,1))
    term = y * term1 + (1 - y) * term2
    J = -((np.sum(term))/m)
    return J

def grad(x, y, theta):
    m , n = x.shape
    theta = theta.reshape((n,1))
    y = y.reshape((m,1))
    sigmoid_x_theta = sigmoid(x.dot(theta))
    grad = ((x.T).dot(sigmoid_x_theta-y))/m
    return grad.flatten()
"""

def gradcost(theta, x, y):
    m = len(y)
    # print m
    n = len(theta)
    # print n
    theta = theta.reshape((n,1))
    # print theta
    # y = y.reshape((m,1))
    term1 = np.log(sigmoid(np.dot(x, theta)))
    # print term1
    term2 = np.log(1-sigmoid(np.dot(x, theta)))
    # print term2
    term1 = term1.reshape((m,1))
    term2 = term2.reshape((m,1))
    term = y * term1 + (1 - y) * term2
    # print term
    J = -((np.sum(term))/m)
    sigmoid_x_theta = sigmoid(x.dot(theta))
    grad = ((x.T).dot(sigmoid_x_theta-y))/m
    print J
    print grad
    return J, grad.flatten()



def sigmoid(z):
    # g = np.zeros(np.shape(np.array(z)))
    # print np.shape([1,2])
    # [k, l] = np.shape(g)
    # for i in range(1, k):
        # for j in range(1, l):
            # g[i,j] = 1/(1+np.exp(-z[i,j]))
    g = 1.0/(1.0+np.exp(-z))
    return g

print sigmoid(10)
print sigmoid(-10)
print sigmoid(0)

# J = cost(x, y, theta)
# grad = grad(x, y, theta)
# print J
# print grad

Result = op.minimize(fun = gradcost, x0 = initial_theta, args = (x, y), method = 'TNC', jac = True)

optimal_theta = Result.x
print optimal_theta
print np.shape(optimal_theta)
print np.shape([1, 45, 85])
print sigmoid(np.dot(optimal_theta, [1, 45, 85]))