__author__ = 'Pratt'

import pylab as pl
import numpy as np


x1_lists = []
x2_lists = []
y_lists = []

with open('ex1data2.txt') as f:
    for line in f:
        # inner_list = [elt.strip() for elt in line.split(',')]
        # in alternative, if you need to use the file content as numbers
        inner_list = [float(elt.strip()) for elt in line.split(',')]
        x1_lists.append(inner_list[0])
        x2_lists.append(inner_list[1])
        y_lists.append(inner_list[2])

print x1_lists
print x2_lists
print y_lists

m = len(x1_lists)
x = np.column_stack([x1_lists,x2_lists])
x.shape = (m,2)
#print x

y = np.array(y_lists)
y.shape = (m,1)
#print y

# x = np.insert(x,0, values=np.ones(m),axis=1)
# print x
theta = np.asarray([0,0,0])
theta.shape = (3,1)

#print theta

def featureNormalize(x):
    x_norm = x
    mu = np.zeros(2)
    sigma = np.zeros(2)

    mu = np.sum(x, axis=0)/m
    sigma = (np.sum((x-mu)**2/(m-1), axis=0))**0.5

    print mu
    print sigma

    x_norm = (x - mu)/sigma
    return x_norm, mu, sigma


[x, mu, sigma] = featureNormalize(x)
x = np.insert(x,0, values=np.ones(m),axis=1)
print x

theta = np.asarray([0,0,0])
theta.shape = (3,1)

# print theta

numIterations = 400
alpha = 0.1
def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    print xTrans
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        # print hypothesis
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
    return theta

theta = gradientDescent(x, y, theta, alpha, m, numIterations)

ytest1 = np.asarray([1650, 3])

ytest = np.insert((ytest1-mu)/sigma, 0, 1, axis=0)

ytest.shape = (1,3)
predict1 = np.dot(ytest, theta)

print predict1

print theta