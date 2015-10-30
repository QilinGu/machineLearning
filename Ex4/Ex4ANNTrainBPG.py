__author__ = 'Pratt'


import scipy.io as spio
import numpy as np
import scipy.optimize as op

matdata = spio.loadmat('ex4data1.mat')
datax = matdata.get('X')
datay = matdata.get('y')
X = np.array(datax)
[m, n] = np.shape(X)

y = np.array(datay)
# print y
for j in range(0, m):
     if (y[j]==10): y[j]=0
#print y
input_layer_size  = 400
hidden_layer_size = 25
num_labels = 10

mattheta = spio.loadmat('ex4weights.mat')
dataxTheta1 = mattheta.get('Theta1')
dataxTheta2 = mattheta.get('Theta2')
Theta1_t = np.array(dataxTheta1)
Theta2_t = np.array(dataxTheta2)
# print Theta1_t, Theta2_t
params = np.concatenate([Theta1_t.ravel(), Theta2_t.ravel()])
# print params

def sigmoid(z):
    # g = scipy.special.expit(z)
    g = 1.0/(1.0+np.exp(-z))
    return g

def sigmoidGradient(z):
    g = np.multiply(sigmoid(z), 1 - sigmoid(z))
    return g

def nnCostFunc(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda):
    Theta1 = nn_params[0:hidden_layer_size*(input_layer_size+1)]
    Theta1 = Theta1.reshape(hidden_layer_size, input_layer_size+1)
    # Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):hidden_layer_size*(input_layer_size+1)+(hidden_layer_size+1)*num_labels]
    Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):]
    Theta2 = Theta2.reshape(num_labels, hidden_layer_size+1)
    m, n = np.shape(X)

    Xhat = np.column_stack([np.ones(m), X])

    Yhat = np.zeros(num_labels)

    k, l = np.shape(Theta1)
    o, p = np.shape(Theta2)
    # Ahat = np.zeros(k+1)

    Del1 = np.zeros(k*l)
    Del1 = Del1.reshape(k,l)
    Del2 = np.zeros(o*p)
    Del2 = Del2.reshape(o,p)

    J = 0
    Theta1_grad = np.zeros(hidden_layer_size*(input_layer_size+1))
    Theta1_grad = Theta1_grad.reshape(hidden_layer_size, input_layer_size+1)
    Theta2_grad = np.zeros(num_labels*(hidden_layer_size+1))
    Theta2_grad = Theta2_grad.reshape(num_labels, hidden_layer_size+1)

    for i in range(0,m):

        A = sigmoid(Theta1.dot(Xhat[i,:].transpose()))

        Ahat = np.zeros(hidden_layer_size+1)
        Ahat[0] = 1
        Ahat[1:] = A[:]
        # print A, Ahat

        Z = sigmoid(Theta2.dot(Ahat).transpose())

        Yhat = np.zeros(num_labels)
        # for c in range (1, num_labels+1):
        for c in range (0, num_labels):
            # if y[i]==c: Yhat[c-1]=1
            if y[i]==c: Yhat[c]=1
            # elif y[i]==num_labels: Yhat[num_labels-1]=1


        # print Yhat, Z
        J += (1.0/m)*(-(np.log(Z)).dot(Yhat) - (np.log(np.ones(num_labels).transpose()-Z)).dot(np.ones(num_labels)-Yhat))
        # print J

        del3 = Z.transpose()-Yhat

        del3 = del3.reshape(num_labels,1)

        #print np.shape(Theta2.transpose().dot(del3))
        #print np.shape(np.r_[1,Theta1.dot(Xhat[i,:].transpose())])

        del2 = np.multiply(Theta2.transpose().dot(del3),sigmoidGradient(np.r_[1, Theta1.dot(Xhat[i,:].transpose())]).reshape(hidden_layer_size+1,1))

        #print np.shape(del2)

        del2 = del2[1:]

        #print np.shape(del2)

        del2 = del2.reshape(hidden_layer_size,1)

        Ahat = Ahat.reshape(k+1,1)

        Del2 = Del2 + del3.dot(Ahat.transpose())

        # print np.shape(Xhat[i,:])

        # Xvalue = np.zeros(input_layer_size+1)
        Xvalue = Xhat[i,:]
        Xvalue = Xvalue.reshape(1, input_layer_size+1)
        #print np.shape(Xvalue)

        Del1 = Del1 + del2.dot(Xvalue)


    J += (lamda/(2.0*m))*(np.sum(np.multiply(Theta1,Theta1)) + np.sum(np.multiply(Theta2,Theta2)))
    # print J

    Theta1_grad[:, 0:1] = (1.0/m)*Del1[:, 0:1]
    Theta1_grad[:, 1:] = (1.0/m)*Del1[:, 1:] + (lamda/m)*(Theta1[:, 1:])

    Theta2_grad[:, 0:1] = (1.0/m)*Del2[:, 0:1]
    Theta2_grad[:, 1:] = (1.0/m)*Del2[:, 1:] + (lamda/m)*(Theta2[:, 1:])

    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])

    return J, grad

J, grad = nnCostFunc(params, input_layer_size, hidden_layer_size, num_labels, X, y, 3)
print J, grad

def randInitWeights(L_in, L_out):
    W = np.zeros(L_out*(1 + L_in))
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1 + L_in)*2.0*epsilon_init - epsilon_init
    return W

initial_Theta1 = np.array(randInitWeights(input_layer_size, hidden_layer_size))
initial_Theta2 = np.array(randInitWeights(hidden_layer_size, num_labels))

initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()])
lamda = 0.1
Result = op.minimize(fun = nnCostFunc, x0 = initial_nn_params, args = (input_layer_size, hidden_layer_size, num_labels, X, y, lamda), method = 'TNC', jac = True)
tr_params = Result.x

Theta1_tr = tr_params[0:hidden_layer_size*(input_layer_size+1)]
Theta1_tr = Theta1_tr.reshape(hidden_layer_size, input_layer_size+1)
# Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):hidden_layer_size*(input_layer_size+1)+(hidden_layer_size+1)*num_labels]
Theta2_tr = tr_params[hidden_layer_size*(input_layer_size+1):]
Theta2_tr = Theta2_tr.reshape(num_labels, hidden_layer_size+1)

def predict(Theta1, Theta2, X):
    m,n = np.shape(X)
    num_labels, khat = np.shape(Theta2)

    p = np.zeros(m)

    h1 = sigmoid(np.column_stack([np.ones(m), X]).dot(Theta1.transpose()))
    h2 = sigmoid(np.column_stack([np.ones(m), h1]).dot(Theta2.transpose()))

    p = np.argmax(h2, axis=1)
    return p

pred = predict(Theta1_tr, Theta2_tr, X)

c = 0.0
for i in range(0,m):
    # if (pred[i]==y[i]-1): c += 1.0
    if (pred[i]==y[i]): c += 1.0

print c
print 100.0*(c/m)



