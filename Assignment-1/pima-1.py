import pandas as pd
import numpy as np
pima= pd.read_csv('pima-id.csv',header=None)
pima=np.array(pima)
print "shape",pima.shape
pima[np.where(pima[:,0]>8),0] = 8

pima[np.where(pima[:,7]<=30),7] = 1
pima[np.where((pima[:,7]>30) & (pima[:,7]<=40)),7] = 2
pima[np.where((pima[:,7]>40) & (pima[:,7]<=50)),7] = 3
pima[np.where((pima[:,7]>50) & (pima[:,7]<=60)),7] = 4
pima[np.where(pima[:,7]>60),7] = 5

pima[:,:8] = pima[:,:8]-pima[:,:8].mean(axis=0)
pima[:,:8] = pima[:,:8]/pima[:,:8].var(axis=0)

# Split into training, validation, and test sets

order = range(np.shape(pima)[0])
#print order
#print "after shuffle"
np.random.shuffle(order)
#print iris[[0,1],:]
#print iris[order,:]
pima = pima[order,:]
target = pima[:,8:]
train = pima[::2,0:8]
print train.shape
traint = target[::2]
valid = pima[1::5,0:8]
print valid.shape
validt = target[1::5]
test = pima[3::4,0:8]
print test.shape
testt = target[3::4]

print train.max(axis=0), train.min(axis=0)

# Train the network

import mlp
net = mlp.mlp(train,traint,10,outtype='logistic')
print "done"
e=0
aa= []
bb = []
cc =[]
for i in [0.001,0.003,0.01,0.03,0.1,0.3]:
	for it in [1000,2500,5000]:
		temp=e
		net.earlystopping(train,traint,valid,validt,i)
		e = net.mlptrain(train,traint,i,it)
		ll = net.confmat(train,traint)
		aa.append(i)
		bb.append(it)
		cc.append(e)
ind = cc.index(min(cc))
print ind,aa[ind],bb[ind],cc[ind]
e = net.mlptrain(train,traint,aa[ind],bb[ind])
ll = net.confmat(test,testt)
print ll

