__author__ = 'Pratt'

import scipy.io as spio
import numpy as np
import scipy.optimize as op

matmovies = spio.loadmat('ex8_movies')
# print matmovies

dataY = matmovies.get('Y')
Y = np.array(dataY)
num_movies, num_users = np.shape(Y)

dataR = matmovies.get('R')
R = np.array(dataR)
print Y, R

matparams = spio.loadmat('ex8_movieParams')
# print matparams

dataX = matparams.get('X')
X = np.array(dataX)
num_movies, num_features = np.shape(X)
dataT = matparams.get('Theta')
Theta = np.array(dataT)
num_users, num_features = np.shape(Theta)
print X, Theta
"""
def cofiCostFunc(X, Theta, Y, R, num_users, num_movies, num_features, lamda):

    # J = 0
    X_grad = np.zeros(num_movies)
    Theta_grad = np.zeros(num_users)

    J = 0.5*np.sum(np.multiply(R, X.dot(Theta.transpose())-Y)**2.0)+ (lamda/2.0)*np.sum(np.multiply(Theta, Theta)) + (lamda/2.0)*np.sum(np.multiply(X, X))

    X_grad = np.multiply(R, X.dot(Theta.transpose())-Y).dot(Theta) + lamda*X
    Theta_grad = np.multiply(R, X.dot(Theta.transpose())-Y).transpose().dot(X) + lamda*Theta

    return J, X_grad, Theta_grad

J, X_grad, Theta_grad  = cofiCostFunc(X[0:4, 0:2], Theta[0:3, 0:2], Y[0:4, 0:3], R[0:4, 0:3], 4, 5, 3, 1.5)
print J, X_grad, Theta_grad
"""

def cofiCostFunc(params, Y, R, n_users, n_movies, n_features, lamda):

    # J = 0
    X_grad = np.zeros(n_movies)
    Theta_grad = np.zeros(n_users)

    X = params[0:n_movies*n_features]
    X = X.reshape(n_movies, n_features)
    Theta = params[n_movies*n_features:n_features*(n_movies+n_users)]
    Theta = Theta.reshape(n_users, n_features)

    J = 0.5*np.sum(np.multiply(R, X.dot(Theta.transpose())-Y)**2.0)+ (lamda/2.0)*np.sum(np.multiply(Theta, Theta)) + (lamda/2.0)*np.sum(np.multiply(X, X))

    X_grad = np.multiply(R, X.dot(Theta.transpose())-Y).dot(Theta) + lamda*X
    Theta_grad = np.multiply(R, X.dot(Theta.transpose())-Y).transpose().dot(X) + lamda*Theta

    params_grad = np.concatenate([X_grad, Theta_grad])

    return J, params_grad

iparams = np.concatenate([X[0:5, 0:3].ravel(), Theta[0:4, 0:3].ravel()])

J, params_grad  = cofiCostFunc(iparams, Y[0:5, 0:4], R[0:5, 0:4], 4, 5, 3, 1.5)
# print J, params_grad

idx_lists = []
movie_lists = []

with open('movie_ids.txt') as f:
    for line in f:
        # inner_list = [elt.strip() for elt in line.split(',')]
        # in alternative, if you need to use the file content as numbers
        inner_list = [(elt.strip()) for elt in line.split(' ')]
        idx_lists.append(inner_list[0])
        movie_lists.append(inner_list[1:])

print idx_lists
print movie_lists

my_ratings = np.zeros(num_movies)
my_ratings_R = np.zeros(num_movies)

my_ratings[0] = 4
my_ratings[97]= 2
my_ratings[6] = 3
my_ratings[11]= 5
my_ratings[53] = 4
my_ratings[63]= 5
my_ratings[65]= 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354]= 5

Y = np.column_stack([my_ratings, Y])

for i in range(0, num_movies):
    if (my_ratings[i]>0): my_ratings_R[i] = 1
    else: my_ratings_R[i] = 0

R = np.column_stack([my_ratings_R, R])

def normalizeRatings(Y, R):
    [m, n] = np.shape(Y)
    Ymean = np.zeros(m)
    Ymean = Ymean.reshape(m,1)
    Ynorm = np.zeros(m*n)
    Ynorm = Ynorm.reshape(m,n)
    for i in range (0,m):
        Yisum = 0.0
        Uicount = 0.0
        for j in range(0,n):
            if R[i,j]==1: Yisum += Y[i,j]; Uicount += 1.0
        Ymean[i] = Yisum/Uicount
        Ynorm[i,:] = np.multiply(R[i,:],Y[i,:] - Ymean[i])

    return Ymean, Ynorm

Ymean, Ynorm = normalizeRatings(Y, R)
print Ymean, np.shape(Ymean)

num_features = 10

X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users+1, num_features)
print X, Theta

initial_params = np.concatenate([X.ravel(),Theta.ravel()])
print np.shape(initial_params)
lamda = 10

Result = op.minimize(fun = cofiCostFunc, x0 = initial_params, args = (Y, R, num_users+1, num_movies, num_features, lamda), method = 'TNC', jac = True)
params = Result.x
# params = initial_params
# print params, np.shape(params)

Xfinal = params[0:num_movies*num_features]
Xfinal = Xfinal.reshape(num_movies, num_features)
Thetafinal = params[num_movies*num_features:num_features*(num_movies+num_users+1)]
Thetafinal = Thetafinal.reshape(num_users+1, num_features)

print Xfinal, Thetafinal

p = Xfinal.dot(Thetafinal.transpose())
print np.shape(p)


my_predictions = p[:,0:1] + Ymean
print my_predictions, np.shape(my_predictions)
sidx = np.argsort(my_predictions, axis=0)
print sidx, np.shape(sidx)

print "\nPredicted Ratings\n"

for i in range (1,11):
    print movie_lists[sidx[num_movies-i]], my_predictions[sidx[num_movies-i]]

print "\nMy Original Ratings\n"

for j in range (0, num_movies):
    if (my_ratings[j]>0): print movie_lists[j], my_ratings[j]


