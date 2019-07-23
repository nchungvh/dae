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
import pandas as pd
from sklearn.model_selection import KFold
df = pd.read_csv('output.csv', sep=',')

data = df.values


##### tinh acc tren tung cot:
for cols_index in range(11):
    data = np.concatenate((data[:,:cols_index],data[:,(cols_index+1):]), axis = 1)
    classify = []
    for i in range(11):
        a = np.asarray(list(set(data[:, i])))
        classify.append(a)

    num_train = int((len(data)*0.8))
    test_data = data[num_train:,:].copy()
    data  = torch.from_numpy(data).double().cuda()





    Dim = data.size(1)
    #
    Min_Val = np.zeros(Dim)
    Max_Val = np.zeros(Dim)


    # num_train = int((len(data)*0.8))
    # train_data = data[:num_train,:]
    # test_data = data[num_train:,:]

    #Normalize
    for i in range(data.size(1)):
        Min_Val[i] = min(data[:,i])
        Max_Val[i] = max(data[:,i] - min(data[:,i]))
        data[:,i] = (data[:,i] - min(data[:,i])) / (max(data[:,i] - min(data[:,i])) + 1e-6)
    train_data = data[:num_train,:]

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
                nn.Linear(Dim, 8),
                nn.ReLU(True),
                nn.Linear(8,8),
                nn.ReLU(True))
            self.decoder = nn.Sequential(
                nn.Linear(8,8),
                nn.ReLU(True),
                nn.Linear(8,8),
                nn.ReLU(True),
                nn.Linear(8,Dim),
                nn.Sigmoid())

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x
    def MSEloss(out,data,mask):
        loss = torch.mean((out*mask-data*mask)**2)/torch.mean(mask)    
        return loss

    def acc(testout,test_data,mask):
        testout = testout*(Max_Val+1e-6)+Min_Val
        dem = 0
        true = 0
        for i in range(len(mask)):
            for j in range(len(mask[0])):
                if(mask[i,j]==0):
                    dem+=1
                    if(classify[j][np.argmin(abs(testout[i,j]-classify[j]))]==test_data[i,j]):
                        true+=1
        true = float(true)
        return true/dem





    skf = KFold(n_splits=10, random_state=None)
    # X is the feature set and y is the target
    scores = []
    for train_index, test_index in skf.split(train_data):
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
                testloss = MSEloss(testout,test,mask)
                testout = testout.detach().numpy()
                acc1 = acc(testout,test_data,mask)
                print('acc:    ',acc1)
                if(epoch == 0):
                    pretestloss = testloss
                if(testloss > pretestloss):
                    print('acc final:                         ',acc1)
                    scores.append(acc1)
                    break

            else:
                continue
            break
    print('acc:    ',np.asarray(scores).mean())
