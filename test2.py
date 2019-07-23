import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import csv
import random
import numpy as np
import torch
import pdb


with open('CLASSIFICATION.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile, delimiter=';'))
    data = np.asarray(data)
classify = []
for i in range(12):
    sort = []
    b = np.zeros(len(data))
    if(i>3):
        for j in range(len(data)):
            b[j] = int(data[j,i])
    else:
        b = data[:,i]
    a = np.sort(b)
    for j in range(len(a)-1):
        if(j ==0):
            sort.append(a[j])
        if(a[j]<a[j+1]):
            sort.append(a[j+1])
    classify.append(sort)


for i in range(4):
    for j in range(len(data)):
        for index,value in enumerate(classify[i]):
            if(data[j,i]==value):
                data[j,i] = index
with open('CLASSIFICATION.pkl','wb') as f:
	pickle.dump(data,f)
pdb.set_trace()

