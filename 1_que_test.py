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
    return arr / float(arr.max(axis=1)[0])

def compute_loss(true_y, predicted_y, deriv = False):
    if deriv == True:
        return -2*np.subtract(true_y, predicted_y)
    # Compute Mean Square Loss
    return np.mean(np.power(np.subtract(true_y, predicted_y),2))

#---------------- Global Variables ------------------------
in_size = 96
hi_size = 48
out_size = 4

#---------------------- Main ------------------------------
W1 = np.random.normal(0, 1, in_size*hi_size).reshape((in_size, hi_size))
W2 = np.random.normal(0, 1, hi_size*out_size).reshape((hi_size, out_size))

X_vec, Y_vec = get_data(class_name='coast', te_tr='Train', class_index = 0)


pre_activation_l1 = np.dot(X_vec, W1)
pre_activation_l1 = normalize(pre_activation_l1)
activation_l1 = sigmoid(pre_activation_l1)
pre_activation_l2 = np.dot(activation_l1, W2)
activation_l2 = sigmoid(pre_activation_l2)

loss = compute_loss(Y_vec, activation_l2)
print compute_loss(Y_vec, activation_l2, deriv=True).shape
print sigmoid(activation_l2, deriv=True).shape
print activation_l1.shape
d_loss_d_beta = sigmoid(activation_l2, deriv=True)

#print loss
