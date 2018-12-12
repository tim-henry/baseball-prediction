import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import random


datapath = '../../ab6.867/CUM_CONCAT/'
#name = 'CUM_CONCAT_SeasAvgPlayers_2010_2017'
#name = 'CUM_CONCAT_MovAvgPlayers_2010_2017'
name = 'CUM_CONCAT_ExpWeiAvgPlayers10_2010_2017'
#name = 'CUM_CONCAT'
#name = 'players_mixture_num_degrees3'
#name = 'players_pca_num_degrees3'
#name = 'players_mixture_num_degrees6'

class Net(nn.Module):
    def __init__(self, inputSize = 22):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(inputSize,100)
        self.fc2 = nn.Linear(100, 10)
        #self.fc3 = nn.Linear(50, 10)
        #self.fc4 = nn.Linear(50,10)
        self.fc5 = nn.Linear(10,1)
        self.sp = nn.Softplus()

        self.dropout = nn.Dropout(0.90) 

    def forward(self, x):
        x = F.relu(self.fc1(x)) #F.relu
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        #x = F.relu(self.fc3(x))
        #x = self.dropout(x)
        #x = F.relu(self.fc4(x))
        #self.dropout(x)
        x = self.fc5(x)

        return(torch.sigmoid(x))


def train(net, data, targets, optimizer, epochs = 10000):
    #asfljhashg
    net.train()
    criterion = nn.MSELoss()

    for k in range(0,epochs + 1):
        optimizer.zero_grad()   # zero the gradient buffers
        output = net(data)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()    # Does the update
        if k%1000 ==0:
            print('Iteration ' + str(k) + ' Loss: ' + str(loss.item()))

            out = output
            out = torch.round(out)
            out = out.detach().numpy()
            out = out.T[0]
            y = targets.numpy().T[0]
            num_correct = np.sum(y == out)
            accuracy = num_correct / len(y)
            print('       ' + ' Accuracy: ' + str(accuracy))
        if k == epochs - 1:
            print('Train Loss: ' + str(loss.item()))

def test(net, data, targets):
    criterion = nn.MSELoss()
    output = net(data)
    loss = criterion(output, targets)
    print('***Test Loss: ' + str(loss.item()))
    return(output)



def dropLowGames(df, colStart = 33, threshold = 5):
    # colnames = df.columns.values[colStart:]
    new_df = df.iloc[:, colStart:]

    print('Thresholding')

    new_df = df.drop([col for col, val in new_df.iloc[:, colStart:].abs().sum().iteritems()
             if val < threshold], axis=1)

    numPlayersRemoved = (len(df.columns) - (len(new_df.columns)))


    # for k, col in enumerate(colnames):
    #     column = df[col].values
    #     #print(col)
    #     if np.sum(np.abs(column)) < threshold:
    #         df = df.drop([col], axis = 1)
    #         numPlayersRemoved = numPlayersRemoved + 1

    print('Number of Players Removed: ' + str(numPlayersRemoved))
    return(new_df)


def training_proceedure(lr = 0.01, epochs = 2000, name = name, kfolds = 2, TtoV = 3, datapath = datapath, playerThreshhold = 5, weight_decay = 1e-3):

    df = pd.read_csv(datapath + name + '.csv')
    df = df.drop(['Season'], axis = 1)
    df = dropLowGames(df = df, threshold = playerThreshhold)
    print('inputSize:' + str(df.shape[1]))
    #df.to_pickle(datapath + name + '.pkl')
    trainSize =  int(np.ceil(TtoV / (TtoV +1) * df.shape[0]))
    testSize = int(df.shape[0] - trainSize)


    dropList = ['cum_Balks','cum_intentionalWalks','cum_putouts']


    #trainSize = 200
    #testSize = 50

    for fold in range(0,kfolds):
        #train_idx  = random.sample(range(0, df.shape[0]), trainSize)
        #test_idx = [v for i, v in enumerate(range(0,df.shape[0])) if i not in train_idx]
        train_idx = list(range(0,int(np.ceil(0.8 * df.shape[0]))))
        test_idx = list(range(train_idx[-1],int(df.shape[0])))
        print('Indices Generated')


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
        net.train()
        optimizer = optim.Adam(net.parameters(), weight_decay = weight_decay)
        #optimizer = optim.RMSProp(net.paramerters(dsofn;sdfj)) or SGD

        #print('training')
        #print('X dims:' + str(dataX.shape))
        #print('Y dims:' + str(dataY.shape))
        print('Training...')
        train(net = net, data = dataX, targets = dataY, optimizer = optimizer, epochs = epochs)

        #dataX = np.array(pd.read_csv(datapath + name + '.csv').drop(['isWin'], axis = 1).drop(dropList, axis =1).values[test_idx,:]).astype(float)
        #dataY = np.array(pd.read_csv(datapath + name + '.csv')['isWin'][test_idx]).astype(float)
        dataX = np.array(df.drop(['isWin'], axis = 1).drop(dropList, axis =1).values[test_idx,:]).astype(float)
        dataY = np.array(df['isWin'][test_idx]).astype(float)
        dataX = torch.Tensor(dataX)#.float()
        dataY = dataY[None].T
        y = dataY
        dataY = torch.Tensor(dataY)#.float()
        net.eval()
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


'''
solid:
200 
0.9 dropout each layer
layers 100 / 10 / 1
test 57
(threshold 50 doesn't change much)
-------------
400 epochs
0.95 dropout each layer
layers 100 / 20  / 1
test 56
'''


'''
Moving Average 15
200 epochs
0.9 dropout
layers 100 /dropout/ 10 / 1
test 55.7
(threshold 50 doesn't change much)


Thresholding
Number of Players Removed: 576
inputSize:1246
Indices Generated
Training...
Epoch 0 Loss: 0.2541823387145996
        Accuracy: 0.4977102732407266
Epoch 20 Loss: 0.2491927444934845
        Accuracy: 0.5182414898488781
Epoch 40 Loss: 0.24528735876083374
        Accuracy: 0.5571286826438712
Epoch 60 Loss: 0.2404443770647049
        Accuracy: 0.582353839108533
Epoch 80 Loss: 0.2359970510005951
        Accuracy: 0.5998702488169745
Epoch 100 Loss: 0.2317851483821869
        Accuracy: 0.6148297969775607
Epoch 120 Loss: 0.22866787016391754
        Accuracy: 0.6294458861242559
Epoch 140 Loss: 0.22512967884540558
        Accuracy: 0.6389864142878949
Epoch 160 Loss: 0.22170911729335785
        Accuracy: 0.6475728896351702
Epoch 180 Loss: 0.22002045810222626
        Accuracy: 0.651808884139826
Train Loss: 0.2182788848876953
***Test Loss: 0.25438380241394043
***Test Accuracy: 0.5567895580490039
'''


'''
moving average / unorderesd
 / 100 / 10 / 1
 dropout 0.9
 weight_decay 1e-3
 iters 5000
 training: 55
 test: 57
 ------------------------
 moving average / orderesd
 / 100 / 10 / 1
 dropout 0.9
 weight_decay 1e-3
 iters 5000
 training: 56
 test: 55.5

 Iteration 0 Loss: 0.2550833523273468
        Accuracy: 0.49769239025437373
Iteration 1000 Loss: 0.24491485953330994
        Accuracy: 0.5585488891274015
Iteration 2000 Loss: 0.24413560330867767
        Accuracy: 0.5580837894887481
Iteration 3000 Loss: 0.24334731698036194
        Accuracy: 0.5612321562734786
Iteration 4000 Loss: 0.24283769726753235
        Accuracy: 0.5609101642159493
Train Loss: 0.24379172921180725
***Test Loss: 0.2465457171201706
***Test Accuracy: 0.5552375500858615

----------------------
 moving average / orderesd
 / 100 / 10 / 1
 dropout 0.5
 weight_decay 1e-3
 iters 5000
 training: 78
 test: 52.8

 Iteration 0 Loss: 0.2503645718097687
        Accuracy: 0.4956888841186362
Iteration 1000 Loss: 0.17746594548225403
        Accuracy: 0.7511359164251726
Iteration 2000 Loss: 0.16327017545700073
        Accuracy: 0.7738900218239061
Iteration 3000 Loss: 0.15997707843780518
        Accuracy: 0.7807234088225824
Iteration 4000 Loss: 0.15813791751861572
        Accuracy: 0.7821902615291045
Train Loss: 0.15713895857334137
***Test Loss: 0.284803569316864
***Test Accuracy: 0.5280480824270177
-----------------------------
 moving average / orderesd
 / 100 / 10 / 1
 dropout 0.9
 weight_decay 1e-2
 iters 10000
 training: 
 test: 
 converges in two seconds
 -----------------------------
  moving average / orderesd
  / 100 / 50 / 10 / 1
   dropout 0.9
    weight_decay 1e-4
    iters 1000

Thresholding
Number of Players Removed: 570
inputSize:1252
Indices Generated
Training...
Iteration 0 Loss: 0.256178617477417
        Accuracy: 0.4998390039712354
Iteration 1000 Loss: 0.23106715083122253
        Accuracy: 0.5904261028227971
Iteration 2000 Loss: 0.22261376678943634
        Accuracy: 0.6067045901756646
Iteration 3000 Loss: 0.2193688154220581
        Accuracy: 0.6136095309649029
Iteration 4000 Loss: 0.21827976405620575
        Accuracy: 0.615147937461987
Train Loss: 0.2181186079978943
***Test Loss: 0.2989087402820587
***Test Accuracy: 0.5314825414997137
'''


#def main():
#    #slkfdn
#    df = pd.read_csv(datapath + name + '.csv')


if __name__ == '__main__':
    training_proceedure(weight_decay = 5e-3,lr = 0.01, name = name, datapath = datapath, epochs = 10000, kfolds = 1, playerThreshhold = 50)



