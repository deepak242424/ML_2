import numpy as np

temp = np.ones(96*48).reshape((96,48))

def exp(temp):
    return 1/(1 + np.exp(- temp))

def normalize(arr):
    return arr / float(arr.max(axis=1)[0])

test1 = np.arange(9)
test1 = np.asarray(test1).reshape((3,3))
#print temp

t3= np.asarray([[.105,.106,.451],[.105,.106,.451]])
#print exp(t3)


abc = np.arange(4)
bcd = np.ones(4)*2
print abc, bcd
#print np.mean(np.power(np.subtract(abc, bcd),2))
print abc*bcd