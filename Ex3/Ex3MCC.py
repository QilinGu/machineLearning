__author__ = 'Pratt'

import scipy.io as spio
import numpy as np
import scipy.optimize as op
import sklearn.multiclass as sklmc
import sklearn.svm as sklsvm

matdata = spio.loadmat('ex3data1.mat')
print matdata

"""
import numpy as np
import h5py
f = h5py.File('ex3data1.mat','r')
data1 = f.get('data/X')
X = np.array(data1) # For converting to numpy array
"""

datax = matdata.get('X')
datay = matdata.get('y')
X = np.array(datax)
[m, n] = np.shape(X)
print m, n, X

# X = np.array(np.column_stack([np.ones(m),X]))

y = np.array(datay)
# for j in range(0, m):
#     if (y[j]==10): y[j]=0
# print X
# print np.shape(X)
# print y
# print np.shape(y)
"""
p = sklmc.OneVsRestClassifier(sklsvm.LinearSVC(random_state=0)).fit(X, y).predict(X)
# print p
# print y
c=0.0
for j in range (0,m):
    if (p[j]==y[j]): c+=1.0

print c
print 100*(c/m)
"""
theta = np.zeros(n+1)
theta.shape = (n+1,1)
initial_theta = theta

num_labels = 10
lamda = 0

def sigmoid(z):
    g = 1.0/(1.0+np.exp(-z))
    return g

def gradcost(theta, x, y, lamda):

    # lamda = 0
    J = 0.0
    m = len(y)
    # print m
    n = len(theta)-1
    # print n
    theta = theta.reshape((n+1,1))
    grad = np.zeros(n+1)
    grad = grad.reshape(n+1,1)
    # print n
    # print np.shape(theta)
    # print theta
    thetat = theta

    thetat = np.delete(thetat, np.s_[0], axis=0)

    thetat = thetat.reshape((n,1))
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
    # print np.shape(term)
    # print term
    J = -((np.sum(term))/m) + term3
    sigmoid_x_theta = sigmoid(x.dot(theta))
    grad = ((x.T).dot(sigmoid_x_theta-y))/m
    thetat = np.array(np.row_stack([0.0, thetat]))
    # print thetat
    # print thetat.shape
    grad = grad + (lamda/m)*thetat
    # print J
    # print grad
    return J, grad

def onevsall(X, y, num_labels):

    [m, n] = np.shape(X)
    all_theta = np.zeros(num_labels*(n+1))
    all_theta = all_theta.reshape(num_labels, n+1)

    X = np.array(np.column_stack([np.ones(m),X]))

    for c in range (1, num_labels+1):
        initial_theta = np.zeros(n+1)
        initial_theta = initial_theta.reshape(n+1, 1)
        Yf = np.zeros(m)
        Yf = Yf.reshape(m,1)
        for j in range(0,m):
            if (y[j]==c): Yf[j]=1
        Result = op.minimize(fun = gradcost, x0 = initial_theta, args = (X, y==c, lamda), method = 'TNC', jac = True)
        # options={'maxIter': 50, 'disp': True})
        # print Result
        all_theta[c-1,:] = Result.x

    return all_theta

all_theta = onevsall(X, y, num_labels)
# print all_theta

def predictonevsall(all_theta, X):
    m, n = np.shape(X)
    # num_labels, g = np.shape(all_theta)
    p = np.zeros(m)
    # p = p.reshape(m,1)

    X = np.array(np.column_stack([np.ones(m),X]))

    print (np.dot(all_theta, X.transpose())).transpose()

    for i in range(0,m):
        p[i] = (np.argmax(np.dot(all_theta, X[i,:].transpose()), axis=0))
    return p


p = predictonevsall(all_theta, X)
print p
print y
print np.shape(p)
print np.shape(y)
c = 0.0
for i in range(0,m):
    if (p[i]==y[i]-1): c += 1.0
    # elif (p[i]==0 and y[i]==10):
      #  c += 1


print 100.0*(c/m)
