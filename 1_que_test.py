import numpy as np
import cPickle
import math

#--------------------------
# Class_Name | Class_index|
#--------------------------
#  coast     |     0      |
#  forest    |     1      |
#  insidecity|     2      |
#  mountain  |     3      |
#-------------------------|

def get_data(class_name, te_tr, class_index):
    pickle_file = open('./1_data/' + class_name + '_' + te_tr+ '.save', 'rb')
    fea_vecs = cPickle.load(pickle_file)
    pickle_file.close()
    fea_vecs = np.asarray(fea_vecs)
    target = np.zeros((fea_vecs.shape[0], out_size))
    target[:,class_index] = 1
    return fea_vecs, target

def sigmoid(x, deriv = False):
    if deriv == True:
        return x*(1-x)
    return 1/(1+np.exp(-x))

def normalize(arr):
    mean = np.array([np.mean(arr, axis=0)]*arr.shape[0])
    std  = np.array([np.std(arr, axis=0)]*arr.shape[0])
    return (arr-mean)/std

def compute_loss(true_y, predicted_y, deriv = False):
    if deriv == True:
        return -2*np.subtract(true_y, predicted_y)
    # Compute Mean Square Loss
    return np.mean(np.power(np.subtract(true_y, predicted_y),2))

#---------------- Global Variables ------------------------
in_size = 96
hi_size = 48
out_size = 4
lrate = .001
n_epochs = 500

#---------------------- Main ------------------------------
W1 = np.random.normal(0, 1, in_size*hi_size).reshape((in_size, hi_size))
W2 = np.random.normal(0, 1, hi_size*out_size).reshape((hi_size, out_size))
np.set_printoptions(suppress=True)

X_vec, Y_vec = get_data(class_name='coast', te_tr='Train', class_index = 0)
X, Y = get_data(class_name='forest', te_tr='Train', class_index = 1)
X_vec = np.concatenate((X_vec, X), axis=0)
Y_vec = np.concatenate((Y_vec, Y), axis=0)
X, Y = get_data(class_name='insidecity', te_tr='Train', class_index = 2)
X_vec = np.concatenate((X_vec, X), axis=0)
Y_vec = np.concatenate((Y_vec, Y), axis=0)
X, Y = get_data(class_name='mountain', te_tr='Train', class_index = 3)
X_vec = np.concatenate((X_vec, X), axis=0)
Y_vec = np.concatenate((Y_vec, Y), axis=0)

#def train(X_vec, Y_vec):
for epoch in range(n_epochs):
    pre_activation_l1 = np.dot(X_vec, W1)
    pre_activation_l1 = normalize(pre_activation_l1)
    activation_l1 = sigmoid(pre_activation_l1)
    pre_activation_l2 = np.dot(activation_l1, W2)
    activation_l2 = sigmoid(pre_activation_l2)

    loss = compute_loss(Y_vec, activation_l2)
    print loss
    delta_beta = compute_loss(Y_vec, activation_l2, deriv=True)*sigmoid(activation_l2, deriv=True)

    d_loss_d_beta = np.zeros((activation_l1.shape[0], activation_l1.shape[1], activation_l2.shape[1]))
    for i in range(d_loss_d_beta.shape[0]):
        for j in range(d_loss_d_beta.shape[1]):
            d_loss_d_beta[i][j] = activation_l1[i][j] * delta_beta[i]

    delta_beta_sum = delta_beta.sum(axis = 1)

    d_loss_d_alpha = np.zeros((X_vec.shape[0], X_vec.shape[1], hi_size))
    beta_km = W2.sum(axis = 1)

    sigmoid_deriv_l1 = sigmoid(activation_l1, deriv=True)
    for i in range(d_loss_d_alpha.shape[0]):
        for j in range(d_loss_d_alpha.shape[1]):
            d_loss_d_alpha[i][j] = delta_beta_sum[i]*X_vec[i][j]*beta_km * sigmoid_deriv_l1[i]

    d_loss_d_alpha = d_loss_d_alpha.sum(axis = 0)
    d_loss_d_beta = d_loss_d_beta.sum(axis = 0)

    W1 = W1 - lrate*d_loss_d_alpha
    W2 = W2 - lrate*d_loss_d_beta

    #print d_loss_d_alpha.shape, d_loss_d_beta.shape

test_X_vec, test_Y_vec = get_data(class_name='coast', te_tr='Test', class_index = 0)
test_X, test_Y = get_data(class_name='forest', te_tr='Test', class_index = 1)
test_X_vec = np.concatenate((test_X_vec, test_X), axis=0)
test_Y_vec = np.concatenate((test_Y_vec, test_Y), axis=0)
test_X, test_Y = get_data(class_name='insidecity', te_tr='Test', class_index = 2)
test_X_vec = np.concatenate((test_X_vec, test_X), axis=0)
test_Y_vec = np.concatenate((test_Y_vec, test_Y), axis=0)
test_X, test_Y = get_data(class_name='mountain', te_tr='Test', class_index = 3)
test_X_vec = np.concatenate((test_X_vec, test_X), axis=0)
test_Y_vec = np.concatenate((test_Y_vec, test_Y), axis=0)

pre_activation_l1 = np.dot(test_X_vec, W1)
pre_activation_l1 = normalize(pre_activation_l1)
activation_l1 = sigmoid(pre_activation_l1)
pre_activation_l2 = np.dot(activation_l1, W2)
activation_l2 = sigmoid(pre_activation_l2)

activation_l2 = np.argmax(activation_l2, axis=1)
test_Y_vec = np.argmax(test_Y_vec, axis=1)
print activation_l2
print test_Y_vec
print 1-np.count_nonzero(activation_l2-test_Y_vec)/80.0

