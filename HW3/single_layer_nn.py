# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 21:41:44 2020

@author: benny
"""

import numpy as np
import sys

f = open(sys.argv[1])
f# = open('test.0')
data = np.loadtxt(f)
train = data[:,1:]
trainlabels = data[:,0]

onearray = np.ones((train.shape[0],1))
train = np.append(train,onearray,axis=1)

#print("train=",train)
#print("train shape=",train.shape)

f = open(sys.argv[2])
#f = open('test.0')
data = np.loadtxt(f)
test = data[:,1:]
testlabels = data[:,0]
onearray = np.ones((test.shape[0],1))
test = np.append(test,onearray,axis=1)

rows = train.shape[0]
cols = train.shape[1]



if len(sys.argv)>3:   
    hidden_nodes = int(sys.argv[3])
else:
    hidden_nodes=3

#print(hidden_nodes)

w = np.random.rand(hidden_nodes)
#print("w=",w)

W = np.random.rand(hidden_nodes,cols)
#print("w=",W)

epochs =1000
eta = 0.001
prevobj = np.inf
i = 0



#calculate objective
hidden_layer = np.matmul(train, np.transpose(W))
#print("hidden_layer=",hidden_layer)
#print("hidden_layer shape=",hidden_layer.shape)

sigmoid = lambda x: 1/(1+np.exp(-x))
hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
#print("hidden_layer = ",hidden_layer)
#print("hidden_layer shape", hidden_layer.shape)

output_layer = np.matmul(hidden_layer,np.transpose(w))
#print("output_layer=",output_layer)

obj=np.sum(np.square(output_layer-trainlabels))
#print("obj=",obj)



#gradient descent begin
stop=0
#stop = 0.000001

while(prevobj - obj > 0.000001  and i < epochs):
    prevobj = obj
    
    #print(hidden_layer[0,:].shape,w.shape)
    
    dellw = 0
    for j in range(0,rows):
        dellw += (np.dot(hidden_layer[j,:],np.transpose(w))-trainlabels[j])*hidden_layer[j,:]

    w = w - eta*dellw




    
    dellW=np.zeros(shape=(rows,hidden_nodes))
    for i in range(hidden_nodes):
        dell=0
        for j in range(0,rows):
        
            dell += np.sum(np.dot(hidden_layer[j,:],w)-trainlabels[j])*w[i] * (hidden_layer[j,i])*(1-hidden_layer[j,i])*train[j]
           
       # dellW[i] = dell
        W[i] = W[i]-eta*dell

    
    hidden_layer = np.matmul(train,np.transpose(W))
    
    hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])

    output_layer = (np.matmul(hidden_layer,np.transpose(w)))
    
    obj = np.sum(np.square(output_layer - trainlabels))

    i = i+1


predict_hidden_node = sigmoid(np.matmul(test,np.transpose(W)))
predictions = np.sign(np.matmul(predict_hidden_node,np.transpose(w)))

print(predictions)










