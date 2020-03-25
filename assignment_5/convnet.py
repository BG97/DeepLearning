# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 19:32:23 2020

@author: benny
"""

import numpy as np
import pandas as pd
from scipy import signal
import sys


if len(sys.argv)>1:
    train_dir = sys.argv[1]
    test_dir = sys.argv[2]
else:
    train_dir = 'train'
    test_dir = 'test'

df = pd.read_csv(train_dir+'/data.csv')
names = df['Name'].values
labels = df['Label'].values 

train_data = np.empty((len(labels),3,3), dtype=np.float )

for i in range(0,len(labels)):
    image_matrix = np.loadtxt(train_dir+'/'+names[i])
    train_data[i] = image_matrix

df_test = pd.read_csv(test_dir+'/data.csv')
test_name = df_test['Name'].values
test_label = df_test['Label'].values 

test_data = np.empty((len(test_label),3,3),dtype=np.float)

for i in range(0,len(test_label)):
    test_image_matrix = np.loadtxt(test_dir+'/'+test_name[i])
    test_data[i] = test_image_matrix


sigmoid = lambda x: 1/(1+ np.exp(-x))

c = np.ones((2,2),dtype=np.float)
stride=1
epochs = 1000
eta = 0.1
stop = 0.0001
prevobj = np.inf
obj = 1000000 

output_layer=0

while(prevobj - obj >stop and i<epochs):

    prevobj = obj

    obj = 0 
    sumOfp = 0
    for i in range(0,len(labels)):
    
        hidden_layer = signal.convolve2d(train_data[i],c, mode='valid')
        sqrtf = (sum(sum(sigmoid(hidden_layer))))/4 - labels[i]
        
        
        
        #temp = np.ones((2,2),dtype=np.float)
        
        sumOfcr=0
        for l in range(len(train_data[i])-1):
            for m in range(len(train_data[i][l])-1):
                sumOfcr=0
                for n in range(2):
                    for o in range(2):
                        cr = sigmoid(hidden_layer[n][o]) *(1 - sigmoid(hidden_layer[n][o]))*train_data[i][l+n*stride][m+o*stride]
                        sumOfcr += cr
                dell = (sqrtf * (sumOfcr))/2
                c[l][m] -= eta*dell
        output_layer = (sum(sum(hidden_layer)))/4
        obj += (output_layer - labels[i])**2

    
    #print("obj=",obj)

#print("C=",c)

for i in range(0,len(test_label)):
    hidden_layer = signal.convolve2d(test_data[i],c, mode='valid')
    for j in range(2):
        for k in range(2):
            hidden_layer[j][k] = sigmoid(hidden_layer[j][k])
    output_layer = (sum(sum(hidden_layer)))/4
    if (output_layer < 0.5):
        print(-1)
    else:
        print(1)