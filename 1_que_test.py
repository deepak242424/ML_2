import numpy as np
import cPickle
import math

def get_data(class_name, te_tr):
    pickle_file = open('./1_data/' + class_name + '_' + te_tr+ '.save', 'rb')
    fea_vecs = cPickle.load(pickle_file)
    pickle_file.close()
    return fea_vecs

W1 = np.random.normal(0, 1, 96*48).reshape((96,48))

fea_vecs = get_data(class_name='coast', te_tr='Train')
fea_vecs = np.asarray(fea_vecs)

pre_activation = np.dot(fea_vecs, W1)

#   activation = math.tanh(pre_activation, )

print pre_activation.shape, activation


