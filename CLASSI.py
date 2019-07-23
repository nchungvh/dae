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
from sklearn.model_selection import KFold


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
np.save('CLASIFICATION.npy',data)
pdb.set_trace()




data = torch.from_numpy(np.asarray(data))

#Normalize
Dim = data.size(1)

Min_Val = np.zeros(Dim)
Max_Val = np.zeros(Dim)

for i in range(Dim):
    Min_Val[i] = min(data[:,i])
    data[:,i] = data[:,i] - min(data[:,i])
    Max_Val[i] = max(data[:,i])
    data[:,i] = data[:,i] / (max(data[:,i]) + 1e-6)

num_train = int((len(data)*0.8))
train_data = data[:num_train,:]
test_data = data[num_train:,:]

 


def add_noise(inputs):
    mask = torch.ones(inputs.size(0),inputs.size(1))/5
    mask1 = torch.zeros(inputs.size(0),inputs.size(1))
    for i in range(inputs.size(0)):
        for j in range(inputs.size(1)):
            if(random.uniform(0,1)>0.2):
                mask[i][j] = inputs[i][j]
                mask1[i][j] = 1
    return mask,mask1

batch_size = 128
learning_rate = 1e-3

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(Dim, 24),
            nn.ReLU(True),
            nn.Linear(24,16),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(16,24),
            nn.ReLU(True),
            nn.Linear(24,24),
            nn.ReLU(True),
            nn.Linear(24,Dim),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
def MSEloss(out,data,mask):
    loss = torch.mean((out*mask-data*mask)**2)/torch.mean(mask)    
    return loss

def RMSE(testout,test_data,mask):


skf = KFold(n_splits=10, random_state=None)
# X is the feature set and y is the target
for train_index, test_index in skf.split(train_data):
    scores = []
    model = autoencoder()
    optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

    train = train_data[train_index]
    test = train_data[test_index]
    dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)

    for epoch in range(20000):
        for data in dataloader:
            noisy_data,mask = add_noise(data)
            noisy_data = Variable(torch.FloatTensor(noisy_data))
            data = data.type(torch.FloatTensor)
            output = model(noisy_data)
            output = output.type(torch.FloatTensor)
            loss1 = MSEloss(output,data,mask)
            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()

            noisy_test,mask = add_noise(test)
            noisy_test = Variable(torch.FloatTensor(noisy_test))
            test = test.type(torch.FloatTensor)
            testout = model(noisy_test)
            testout = testout.type(torch.FloatTensor)
            #pdb.set_trace()
            testloss = MSE(testout,test,mask)
            if(epoch == 0):
                pretestloss = testloss
            if(testloss > pretestloss):
                loss = RMSE(testout,test_data,mask)
                scores.append(loss)
                break

        else:
            continue
        break
    print('acc:    ',scores.mean())
