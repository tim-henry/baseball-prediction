import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import random


datapath = '../../ab6.867/CUM_CONCAT/'
name = 'CUM_CONCAT_SeasAvg_2005_2017'
#name = 'CUM_CONCAT'

class Net(nn.Module):
    def __init__(self, inputSize = 22):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(inputSize,20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10,1)
        self.sp = nn.Softplus()

        self.dropout = nn.Dropout(0.5) #NOT REALLY SURE HOW TO IMPLEMENT WITH TEST

    def forward(self, x):
        x = F.relu(self.fc1(x)) #F.relu
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return(torch.sigmoid(x))


def train(net, data, targets, optimizer, epochs = 10000):
    #asfljhashg
    net.train()
    criterion = nn.MSELoss()

    for k in range(0,epochs):
        optimizer.zero_grad()   # zero the gradient buffers
        output = net(data)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()    # Does the update
        if k%500 ==0:
            print('Epoch ' + str(k) + ' Loss: ' + str(loss.item()))
        if k == epochs - 1:
            print('Train Loss: ' + str(loss.item()))

def test(net, data, targets):
    criterion = nn.MSELoss()
    output = net(data)
    loss = criterion(output, targets)
    print('***Test Loss: ' + str(loss.item()))
    return(output)

def training_proceedure(lr = 0.01, epochs = 2000, name = name, kfolds = 2, TtoV = 3, datapath = datapath):

    df = pd.read_csv(datapath + name + '.csv')
    #df.to_pickle(datapath + name + '.pkl')
    trainSize =  int(np.ceil(TtoV / (TtoV +1) * df.shape[0]))
    testSize = int(df.shape[0] - trainSize)
    print('Data Read In')


    dropList = ['cum_Balks','cum_intentionalWalks','cum_putouts']


    #trainSize = 200
    #testSize = 50

    for fold in range(0,kfolds):
        train_idx  = random.sample(range(0, df.shape[0]), trainSize)
        test_idx = [v for i, v in enumerate(range(0,df.shape[0])) if i not in train_idx]


        dataX = np.array(df.drop(['isWin'], axis = 1).drop(dropList, axis =1).values[train_idx,:]).astype(float)
        #dataX = np.array(pd.read_csv(datapath + name + '.csv').drop(['isWin'], axis = 1).values[train_idx,1:]).astype(float)
        #dataY = np.array(pd.read_csv(datapath + name + '.csv')['isWin'][train_idx]).astype(float)
        dataY = np.array(df['isWin'][train_idx]).astype(float)


        inputSize = dataX.shape[1]
        dataX = torch.Tensor(dataX)
        dataY = dataY[None].T
        dataY = torch.Tensor(dataY)

        # create optimizer
        device = torch.device('cpu')
        net = Net(inputSize = inputSize).to(device)
        optimizer = optim.SGD(net.parameters(), lr=lr)

        #print('training')
        #print('X dims:' + str(dataX.shape))
        #print('Y dims:' + str(dataY.shape))
        print('Training...')
        train(net = net, data = dataX, targets = dataY, optimizer = optimizer, epochs = epochs)

        dataX = np.array(pd.read_csv(datapath + name + '.csv').drop(['isWin'], axis = 1).drop(dropList, axis =1).values[test_idx,:]).astype(float)
        dataY = np.array(pd.read_csv(datapath + name + '.csv')['isWin'][test_idx]).astype(float)
        dataX = torch.Tensor(dataX)#.float()
        dataY = dataY[None].T
        y = dataY
        dataY = torch.Tensor(dataY)#.float()
        out = test(net = net, data = dataX, targets = dataY)
        #print(out)
        out = torch.round(out)
        out = out.detach().numpy()
        #print(out)
        #print(y.T)
        #print(out.T)

        y = y.T[0]
        out = out.T[0]

        num_correct = np.sum(y == out)
        accuracy = num_correct / len(y)
        print('***Test Accuracy: ' + str(accuracy))





#def main():
#    #slkfdn
#    df = pd.read_csv(datapath + name + '.csv')


if __name__ == '__main__':
    training_proceedure(lr = 1, name = name, datapath = datapath, epochs = 2000, kfolds = 2)



