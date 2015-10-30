__author__ = 'Pratt'

# import packages
# import datasets

# ex1data1.txt = argv

# txt = open(ex1data1.txt)
# print txt.read

# file_again = raw_input("> ")

# txt_again = open(file_again)

# print txt_again.read()

# stories = datasets.load_sample_stories;

# stories is a matrix containing user's ratings of stories
# 0 to 1, based on % of story listened to

# Assume X is given, matrix of stories' features (categories) --> Content based
# Start with parent categories as features

# Train Theta based on X and stories ratings (user listening history)

# for each user, Sort Theta Stories on descending order of Theta(j)
# Average over the users in each location/city to get playlist rule order

# from sys import argv
# from numpy import loadtxt
import matplotlib
import pylab as pl
import numpy as np

# script, filename = argv

# txt = open(filename)

# print "Here's your file %r:" % filename
# print txt.read()

# print "Type the filename again:"
# file_again = raw_input("> ")

# txt_again = open(file_again)

# print txt_again.read()

x_lists = []
y_lists = []

with open('ex1data1.txt') as f:
    for line in f:
        # inner_list = [elt.strip() for elt in line.split(',')]
        # in alternative, if you need to use the file content as numbers
        inner_list = [float(elt.strip()) for elt in line.split(',')]
        x_lists.append(inner_list[0])
        y_lists.append(inner_list[1])

print x_lists
print y_lists

pl.plot(x_lists, y_lists, 'rx')
pl.show()

m = len(x_lists)
x = np.array(x_lists)
x.shape = (m,1)
print x

y = np.array(y_lists)
y.shape = (m,1)

x = np.insert(x,0, values=np.ones(m),axis=1)
#x.shape = (m,2)
print x
theta = np.asarray([0,0])
theta.shape = (2,1)

print theta

numIterations = 1500
alpha = 0.01

# m denotes the number of examples here, not the number of features
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


ytest1 = np.asarray([1, 3.5])
ytest1.shape = (1,2)

ytest2 = np.asarray([1, 7])
ytest2.shape = (1,2)

predict1 = np.dot(ytest1, theta)
predict2 = np.dot(ytest2, theta)

print predict1
print predict2

print theta
