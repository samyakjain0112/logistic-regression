#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
x=np.array([1,3,5,2,6,7,8,9,4])
y=np.array([0,0,1,0,1,1,1,1,0])
alpha=0.05
def sigmoid(theta,x):
    l=[]
    for i in range(len(x)):
        s=1/(1+np.exp(-theta[0]*x[i][0]-theta[1]*x[i][1]))
        #print(s)
        l.append(s)
    l=np.array(l)
    #print(len(l))
    #print(l.shape)
    #print(np.sum((l-y)*x.T,axis=1))
    return l
theta0=1
theta1=1
X=x.reshape(9,1)
print(x.shape)
X=np.insert(X,0,1,axis=1)
#print(x)
theta=np.array([theta0,theta1])
for i in range(10000):
    theta=theta-(alpha/len(x))*np.sum((sigmoid(theta,X)-y)*X.T,axis=1)
cost=np.sum((alpha/len(x))*((y*np.log(sigmoid(theta,X)))+((np.ones(len(y))-y)*np.log(np.ones(len(y))-sigmoid(theta,X)))))
print(cost)
import matplotlib.pyplot as plt
plt.scatter(x,sigmoid(theta,X))
plt.show()

