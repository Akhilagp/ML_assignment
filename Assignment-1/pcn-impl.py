#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pylab as pl
import numpy as np
import pcn
import pandas as pd


# In[2]:


pima = np.loadtxt('dataset/pima-id.csv',delimiter=',')
# Plot the first and second values for the two classes
indices0 = np.where(pima[:,8]==0)
indices1 = np.where(pima[:,8]==1)

pl.ion()
pl.plot(pima[indices0,0],pima[indices0,1],'go')
pl.plot(pima[indices1,0],pima[indices1,1],'rx')
pl.show()


# In[3]:


print "Output on original data"
p = pcn.pcn(pima[:,:8],pima[:,8:9])
p.pcntrain(pima[:,:8],pima[:,8:9],0.25,100)
p.confmat(pima[:,:8],pima[:,8:9])


# In[4]:


# Various preprocessing steps
pima[np.where(pima[:,0]>8),0] = 8

pima[np.where(pima[:,7]<=30),7] = 1
pima[np.where((pima[:,7]>30) & (pima[:,7]<=40)),7] = 2
pima[np.where((pima[:,7]>40) & (pima[:,7]<=50)),7] = 3
pima[np.where((pima[:,7]>50) & (pima[:,7]<=60)),7] = 4
pima[np.where(pima[:,7]>60),7] = 5


# In[5]:


# Normalize inputs
pima[:,:8] = pima[:,:8]-pima[:,:8].mean(axis=0)
pima[:,:8] = pima[:,:8]/pima[:,:8].var(axis=0)


# In[6]:


# splitting inputs
# odd values for test and even values for training
trainin = pima[::2,:8]
testin = pima[1::2,:8]
traintgt = pima[::2,8:9]
testtgt = pima[1::2,8:9]


# In[7]:


print "Output after preprocessing of data"
p1 = pcn.pcn(trainin,traintgt)


# In[8]:


#vary learning rate to check the best fit
#vary no of iiterations
learning_rate = [0.01,0.03,0.1,0.25,0.3]
iterations = [100, 500, 1000, 2000]
store_accu = []
for lr in learning_rate:
    for it in iterations:
        p1.pcntrain(trainin,traintgt,lr,it)
        accu = p1.confmat(trainin,traintgt)
	print lr,",",it,",",accu
        store_accu.append([accu,lr,it])

# In[9]:

print store_accu
store_accu = np.array(store_accu)


# In[10]:


id = store_accu.max(axis=0)[0]
id = np.where(store_accu[:,0]==id)
print "The values from these indexes correspond to the maximum accuracies: ",
id =  [i for i in np.nditer(id)]
print id
id=id[0]
#print "The highest accuracy obtained from: ",store_accu[id,:]
max_lr,max_it =  np.around(store_accu[id,1],5),store_accu[id,2]
print "To be passed to the final training:\nLearning rate- ",max_lr,"\nNumber of iterations- ",max_it,"\nThe corresponding array index- ",id


# In[11]:


# training the perceptron with the the highest score
p1.pcntrain(trainin,traintgt,max_lr,int(max_it))
p1.confmat(trainin,traintgt)


# In[12]:


#evaluating the test set with the trained perceptron
print p1.confmat(testin,testtgt)


# In[ ]:




