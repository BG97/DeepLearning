# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 21:41:44 2020

@author: benny
"""

import numpy as np
import sys

f = open(sys.argv[1])

#f = open('ion.train.0')
data = np.loadtxt(f)
data_s = data.copy()
train = data[:,1:]
trainlabels = data[:,0]

onearray = np.ones((train.shape[0],1))
train = np.append(train,onearray,axis=1)

mini_batch_size = 32
mini_batch_array = np.random.randint(0,len(data),mini_batch_size)

train_s = data_s[:mini_batch_size,1:]
onearray = np.ones((train_s.shape[0],1))
train_s = np.append(train_s,onearray,axis=1)
trainlabels_s = data_s[:mini_batch_size,0]

#print("train=",train)
#print("train shape=",train.shape)

f = open(sys.argv[2])
#f = open('ion.test.0')
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

#hidden_nodes=4
#print(hidden_nodes)

w = np.random.rand(hidden_nodes)
#print("w=",w)

W = np.random.rand(hidden_nodes,cols)
#print("w=",W)

epochs =10000
eta = 0.01
prevobj = np.inf
k = 0



#calculate objective
hidden_layer = np.matmul(train, np.transpose(W))
#print("hidden_layer=",hidden_layer)
#print("hidden_layer shape=",hidden_layer.shape)

sigmoid = lambda x: 1/(1+np.exp(-x))
hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
#print("hidden_layer = ",hidden_layer)
#print("hidden_layer shape", hidden_layer.shape)
hidden_layer_fullDS = np.matmul(train, np.transpose(W))
hidden_layer_fullDS = np.array([sigmoid(xi) for xi in hidden_layer_fullDS])

output_layer = np.matmul(hidden_layer_fullDS,np.transpose(w))
#print("output_layer=",output_layer)

obj=np.sum(np.square(output_layer-trainlabels))
#print("obj=",obj)


best_w = np.random.rand(hidden_nodes)
best_W = np.random.rand(hidden_nodes, cols)
bestobj = 100000

#gradient descent begin
stop=0
#stop = 0.000001

while(k < epochs):
    #prevobj = obj
    
    #print(hidden_layer[0,:].shape,w.shape)
    mini_batch_array = np.random.randint(0,train.shape[0],mini_batch_size)
    w = best_w
    W = best_W
    dellw = 0
    for j in range(0,mini_batch_size):
        dellw += (np.dot(hidden_layer[mini_batch_array[j],:],np.transpose(w))-trainlabels[mini_batch_array[j]])*hidden_layer[mini_batch_array[j],:]

    w = w - eta*dellw

    #dellW=np.zeros(shape=(rows,hidden_nodes))
    for i in range(hidden_nodes):
        dell=0
        for j in range(0,mini_batch_size):
        
            dell += np.sum(np.dot(hidden_layer[mini_batch_array[j],:],w)-trainlabels[mini_batch_array[j]])*w[i] * (hidden_layer[mini_batch_array[j],i])*(1-hidden_layer[mini_batch_array[j],i])*train[mini_batch_array[j]]           
       # dellW[i] = dell
        W[i] = W[i]-eta*dell

    
    hidden_layer = np.matmul(train,np.transpose(W))
    
    hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])

    output_layer = (np.matmul(hidden_layer,np.transpose(w)))
    
    obj = np.sum(np.square(output_layer - trainlabels))
    if(obj < bestobj):
        bestobj = obj
        best_w = w
        best_W = W

    k= k+1
    #print(k, bestobj)

predict_hidden_node = sigmoid(np.matmul(test,np.transpose(best_W)))
predictions = np.sign(np.matmul(predict_hidden_node,np.transpose(best_w)))

print(predictions)










