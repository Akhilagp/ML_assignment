import pylab as pl
import numpy as np
import pandas as pd
from pylab import *
from numpy import *
import linreg
from scipy import stats
import csv

auto= np.loadtxt('auto-mpg.data',comments='"')
#normalise the data
auto = auto-auto.max(axis=0)/(auto.max(axis=0)-auto.min(axis=0))
trainin=auto[:320,:8]
testin=auto[320:,:8]
traintgt=auto[:320,1:2]
testtgt=auto[320:,1:2]
print "Training shape\n"
print "X_train shape ",trainin.shape," Y_train shape ",traintgt.shape
print "X_test shape ",testin.shape," Y_test shape ", testtgt.shape
def linreg(trainin,traintgt):
	trainin=np.concatenate((trainin,-np.ones((np.shape(trainin)[0],1))),axis=1)
	beta=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(trainin),trainin)),np.transpose(trainin)),traintgt)
	traintgt=np.dot(trainin, beta)
	return beta
beta=linreg(trainin,traintgt)
print "The INITIAL WEIGHTS Chosen ",beta
testin=concatenate((testin,-np.ones((np.shape(testin)[0],1))),axis=1)
testout=dot(testin,beta)
error=sum((testout-testtgt)**2)
print "Cost Function/Error ",error
change = range(np.shape(auto)[0])
np.random.shuffle(change)
auto = auto[change,:]
xtrain = auto[:300,1]
ytrain = auto[:300,0]
xtest = auto[::5,1]
ytest = auto[::5,0]
gradient, intercept, r_value, p_value, std_err = stats.linregress(xtrain,ytrain)
mn=np.min(xtrain)
mx=np.max(xtrain)
x1=np.linspace(mn,mx,10)
y1=gradient*x1+intercept
plt.plot(xtrain,ytrain,'go')
plt.xlabel('cylinders')
plt.ylabel('mpg')
plt.plot(x1,y1,'--r')
plt.show()  
plt.xlabel('cylinders')
plt.ylabel('mpg')
plt.plot(xtest,ytest,'go')
plt.plot(x1,y1,'--r')
plt.show()

