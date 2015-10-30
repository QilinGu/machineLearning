__author__ = 'Pratt'

import scipy.io as spio
import numpy as np
import scipy

matdata = spio.loadmat('ex3data1.mat')
# print matdata

datax = matdata.get('X')
datay = matdata.get('y')
X = np.array(datax)
[m, n] = np.shape(X)
# print m, n, X

y = np.array(datay)
# for j in range(0, m):
#   if (y[j]==10): y[j]=0

# print y


input_layer_size  = 400
hidden_layer_size = 25
num_labels = 10

mattheta = spio.loadmat('ex3weights.mat')
# print mattheta
dataxTheta1 = mattheta.get('Theta1')
dataxTheta2 = mattheta.get('Theta2')
Theta1 = np.array(dataxTheta1)
Theta2 = np.array(dataxTheta2)

# print Theta1, np.shape(Theta1)
# print Theta2, np.shape(Theta2)

def sigmoid(b):
    g = scipy.special.expit(b)
    # g = 1.0/(1.0+np.exp(-B))
    return g


def predictNN(Theta1, Theta2, X):
    m, n = np.shape(X)
    # num_labels, hlplusone = np.shape(Theta2)

    # p = np.zeros(m)
    Xhat = np.array(np.column_stack([np.ones(m),X]))

    # A = (sigmoid(np.dot(Theta1,Xhat.transpose()))).transpose()
    A = (sigmoid(np.dot(Xhat, Theta1.transpose())))
    A = np.array(np.column_stack([np.ones(m),A]))

    Z = (sigmoid(np.dot(A, Theta2.transpose())))

    print Z

    p = np.argmax(Z, axis=1)

    print p, np.shape(p), np.shape(y)

    return p

pred = predictNN(Theta1, Theta2, X)

c = 0.0
for i in range(0,m):
    if (pred[i]==y[i]-1): c += 1.0

print c
print 100.0*(c/m)

