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
# for cols_index in range(11):
#     data = np.concatenate((Data[:,:cols_index],Data[:,(cols_index+1):]), axis = 1)
classify = []
for i in range(12):
    a = np.asarray(list(set(data[:, i])))
    classify.append(a)

num_train = int((len(data)*0.8))
test_data = data[num_train:,:].copy()
datagoc  = torch.from_numpy(data).double()





Dim = datagoc.size(1)
#
Min_Val = np.zeros(Dim)
Max_Val = np.zeros(Dim)


# num_train = int((len(data)*0.8))
# train_data = data[:num_train,:]
# test_data = data[num_train:,:]

#Normalize
for i in range(datagoc.size(1)):
    Min_Val[i] = min(datagoc[:,i])
    Max_Val[i] = max(datagoc[:,i] - min(datagoc[:,i]))
    datagoc[:,i] = (datagoc[:,i] - min(datagoc[:,i])) / (max(datagoc[:,i] - min(datagoc[:,i])) + 1e-6)
train_data = datagoc[:num_train,:]
test_set = datagoc[num_train:,:]
#pdb.set_trace()

def add_noise(inputs):
    mask = torch.ones(inputs.size(0),inputs.size(1))/2
    mask1 = torch.zeros(inputs.size(0),inputs.size(1))
    for i in range(inputs.size(0)):
        for j in range(inputs.size(1)):
            if(random.uniform(0,1)>0.2):
                mask[i][j] = inputs[i][j]
                mask1[i][j] = 1
    return mask,mask1

batch_size = 128
learning_rate = 1e-3

def add_noise_acc(inputs,index):
    inputs = inputs.double()
    mask = torch.zeros(inputs.size(0),inputs.size(1)).double()
    mask[:,index] = 0.5 
    mask1 = torch.ones(inputs.size(0),inputs.size(1)).double()
    mask1[:,index] = 0
    out = (mask1*inputs+mask).float()
    return out,mask1



class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(Dim, 16),
            nn.ReLU(True),
            nn.Linear(16,16),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(16,16),
            nn.ReLU(True),
            nn.Linear(16,Dim),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
def MSEloss(out,data,mask):
    loss = torch.mean((out*mask-data*mask)**2)/torch.mean(mask)    
    return loss

def acc(testout,test_data, index):
    testout = testout[:,index]*(Max_Val[index]+1e-6)+Min_Val[index]
    true = 0
    for j in range(len(test_data)):
        if(classify[index][np.argmin(abs(testout[j]-classify[index]))]==test_data[j,index]):
            true+=1
    true = float(true)
    #pdb.set_trace()
    return true/len(test_data)





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

    for epoch in range(2000):
        count_iters = 0
        for data in dataloader:
            count_iters+= 1

            noisy_data,mask = add_noise(data)
            noisy_data = Variable(torch.FloatTensor(noisy_data))
            data = data.type(torch.FloatTensor)
            output = model(noisy_data)
            output = output.type(torch.FloatTensor)
            loss1 = MSEloss(output,data,mask)
            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()

            #compute MSELoss on valid data

            # noisy_test,mask = add_noise(test)
            # noisy_test = Variable(torch.FloatTensor(noisy_test))
            # test = test.type(torch.FloatTensor)
            # testout = model(noisy_test)
            # testout = testout.type(torch.FloatTensor)
            # testloss = MSEloss(testout,test,mask)
            
            # #compute acc on test_data
            # if(count_iters%5 ==0):
            #     print('epoch: {}  iters: {}    train loss: {}      valid loss: {}     '.format(epoch,count_iters,loss1,testloss))
            
            # if((epoch == 0)&(count_iters==1)):
            #     pretestloss = testloss    #init pretestloss at epoch 0, iter 1
            # if(testloss <= pretestloss):
            #     pretestloss = testloss
            # if(testloss > pretestloss): #compare validloss with pre vlid loss
            for index in range(12):
                noisy_test_set,mask = add_noise_acc(test_set,index)
                noisy_test_set = Variable(torch.FloatTensor(noisy_test_set))
                test_set = test_set.type(torch.FloatTensor)
                test_set_out = model(noisy_test_set)
                test_set_out = test_set_out.type(torch.FloatTensor)
                # test_set_loss = MSEloss(test_set_out,test_set,mask)
                test_set_out = test_set_out.detach().numpy()
                acc1 = acc(test_set_out,test_data,index)

                print('acc final: {}   index:  {}     trainloss: {}                    '.format(acc1,index,loss1))



        else:
            continue
        break
print('acc:    ',np.asarray(scores).mean())
