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

###### data.csv #######################
# with open('CLASSIFICATION.csv', newline='') as csvfile:
#     data = list(csv.reader(csvfile))
#     print(data[4])
# pdb.set_trace()


# data = data[1:]
# for i in range(569):
#     if(data[i][1] == 'M'):
#         data[i][1] = 0
#     else:
#         data[i][1] = 1
#     for j in range(32):
#         data[i][j]= float(data[i][j])
# data = torch.from_numpy(np.asarray(data))

# #Normalize
# data = data[:,1:]
# Dim = data.size(1)

# Min_Val = np.zeros(Dim)
# Max_Val = np.zeros(Dim)

# for i in range(Dim):
#     Min_Val[i] = min(data[:,i])
#     data[:,i] = data[:,i] - min(data[:,i])
#     Max_Val[i] = max(data[:,i])
#     data[:,i] = data[:,i] / (max(data[:,i]) + 1e-6)

# num_train = int((len(data)*0.8))
# train_data = data[:num_train,:]
# test_data = data[num_train:,:]


############## Letter.csv ####################


Data = np.loadtxt("Letter.csv", delimiter=",",skiprows=1)

# Parameters
No = len(Data)
Dim = len(Data[0,:])

# Hidden state dimensions
H_Dim1 = Dim
H_Dim2 = Dim

# Normalization (0 to 1)
Min_Val = np.zeros(Dim)
Max_Val = np.zeros(Dim)

for i in range(Dim):
    Min_Val[i] = np.min(Data[:,i])
    Data[:,i] = Data[:,i] - np.min(Data[:,i])
    Max_Val[i] = np.max(Data[:,i])
    Data[:,i] = Data[:,i] / (np.max(Data[:,i]) + 1e-6)    

#%% Missing introducing
    
#%% Train Test Division    
   
idx = np.random.permutation(No)

Train_No = int(No * 0.8)
Test_No = No - Train_No
Data = torch.from_numpy(np.asarray(Data))
# Train / Test Features
train_data = Data[idx[:Train_No],:]
test_data = Data[idx[Train_No:],:]






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
dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

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
    return torch.sqrt(torch.mean(((1-mask)*testout-(1-mask)*test_data)**2)/torch.mean(1-mask))

model = autoencoder()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

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

    if(epoch%2==0):
        #print(loss1)
        noisy_test,mask = add_noise(test_data)
        noisy_test = Variable(torch.FloatTensor(noisy_test))
        test_data = test_data.type(torch.FloatTensor)
        testout = model(noisy_test)
        testout = testout.type(torch.FloatTensor)
        #pdb.set_trace()
        testloss = MSEloss(testout,test_data,mask)
        print('RMSE:                         ',testloss)

#torch.save(model.state_dict(), './sim_autoencoder.pth')
