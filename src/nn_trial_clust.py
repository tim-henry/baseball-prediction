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
#name = 'CUM_CONCAT'
#name = 'players_mixture_num_degrees3'
#name = 'players_pca_num_degrees3'
name = 'players_mixture_num_degrees6'

class Net(nn.Module):
    def __init__(self, inputSize = 22):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(inputSize,30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 30)
        self.fc4 = nn.Linear(30,10)
        self.fc5 = nn.Linear(10,1)
        self.sp = nn.Softplus()

        self.dropout = nn.Dropout(0.0) #NOT REALLY SURE HOW TO IMPLEMENT WITH TEST

    def forward(self, x):
        x = F.relu(self.fc1(x)) #F.relu
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        self.dropout(x)
        x = self.fc5(x)

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


def training_proceedure(lr = 0.01, epochs = 2000, name = name, kfolds = 2, TtoV = 3, datapath = datapath, playerThreshhold = 5, weight_decay = 1e-4):

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
        optimizer = optim.Adam(net.parameters(), weight_decay = weight_decay )
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


''' UNORDERED
 / 30 / 10 / 1 
 0.5 dropout
 weight_decay 1e-3
 iters: 7000
 Train : 54.5
 Test : 53
-------------------------
 / 30 / 10 / 1 
 0.5 dropout
 weight_decay 1e-3
 iters: 10000
 Train : 54.57
 Test : 53.2
-------------------
 / 30 / 10 / 1 
 0.9 dropout
 weight_decay 1e-4
 iters: 10000
 Train : 54.57
 Test : 53.2

Iteration 0 Loss: 0.27865439653396606
        Accuracy: 0.4992576223196713
Iteration 1000 Loss: 0.2499191164970398
        Accuracy: 0.5020544870688167
Iteration 2000 Loss: 0.2497503161430359
        Accuracy: 0.5073374538172024
Iteration 3000 Loss: 0.2496323436498642
        Accuracy: 0.5150029349815268
Iteration 4000 Loss: 0.24930685758590698
        Accuracy: 0.5154863437036014
Iteration 5000 Loss: 0.24967187643051147
        Accuracy: 0.5122060702323815
Iteration 6000 Loss: 0.24946510791778564
        Accuracy: 0.5151755809536963
Iteration 7000 Loss: 0.24959032237529755
        Accuracy: 0.5108249024550258
Iteration 8000 Loss: 0.24954535067081451
        Accuracy: 0.5133800628431339
Iteration 9000 Loss: 0.24949979782104492
        Accuracy: 0.5161078692034115
Train Loss: 0.249530628323555
***Test Loss: 0.24926318228244781
***Test Accuracy: 0.5312338133222833

 ---------------------
  / 30 / 10 / 1 
 0.9 dropout
 weight_decay 1e-5
 iters: 10000
 Train : 51.77
 Test : 53.36

 Thresholding
Number of Players Removed: 0
inputSize:32
Indices Generated
Training...
Iteration 0 Loss: 0.2769608497619629
        Accuracy: 0.49949932668070857
Iteration 1000 Loss: 0.24993525445461273
        Accuracy: 0.5034356548461725
Iteration 2000 Loss: 0.24977633357048035
        Accuracy: 0.5038845343738131
Iteration 3000 Loss: 0.24944646656513214
        Accuracy: 0.5166603363143538
Iteration 4000 Loss: 0.24933496117591858
        Accuracy: 0.5219433030627395
Iteration 5000 Loss: 0.24954748153686523
        Accuracy: 0.5157971064535064
Iteration 6000 Loss: 0.24933120608329773
        Accuracy: 0.5147267014260557
Iteration 7000 Loss: 0.2495070844888687
        Accuracy: 0.5114464279548359
Iteration 8000 Loss: 0.24951909482479095
        Accuracy: 0.5138634715652084
Iteration 9000 Loss: 0.24933135509490967
        Accuracy: 0.5177307413418045
Train Loss: 0.24956372380256653
***Test Loss: 0.24922314286231995
***Test Accuracy: 0.5336164922821921
----------------------
  / 30 / 10 / 1 
 0.9 dropout
 weight_decay 1e-6
 iters: 10000

'''


'''ORDERED
  / 30 / 10 / 1 
dropout 0.90
weight_decay 1e-3
iters 10000

Thresholding
Number of Players Removed: 0
inputSize:32
Indices Generated
Training...
Iteration 0 Loss: 0.278171569108963
        Accuracy: 0.49841382882299623
Iteration 1000 Loss: 0.24982909858226776
        Accuracy: 0.5089343519357763
Iteration 2000 Loss: 0.24978837370872498
        Accuracy: 0.514566878156157
Iteration 3000 Loss: 0.24969588220119476
        Accuracy: 0.5154085200051793
Iteration 4000 Loss: 0.24976220726966858
        Accuracy: 0.5141784280719928
Iteration 5000 Loss: 0.24975857138633728
        Accuracy: 0.5129159652984592
Iteration 6000 Loss: 0.24976283311843872
        Accuracy: 0.5130454486598472
Iteration 7000 Loss: 0.2496623545885086
        Accuracy: 0.5175126246277353
Iteration 8000 Loss: 0.24960456788539886
        Accuracy: 0.5143079114333808
Iteration 9000 Loss: 0.2497251033782959
        Accuracy: 0.5127541110967241
Train Loss: 0.24960435926914215
***Test Loss: 0.24957457184791565
***Test Accuracy: 0.5311407484138289
------------------------------------------

  / 30 / 10 / 1 
dropout 0.90
weight_decay 1e-4
iters 10000
Iteration 0 Loss: 0.26604241132736206
        Accuracy: 0.49941732487375373
Iteration 1000 Loss: 0.24995753169059753
        Accuracy: 0.5028810047908844
Iteration 2000 Loss: 0.24943609535694122
        Accuracy: 0.5191635374854331
Iteration 3000 Loss: 0.24946415424346924
        Accuracy: 0.518807458241616
Iteration 4000 Loss: 0.24945032596588135
        Accuracy: 0.5185484915188399
Iteration 5000 Loss: 0.24941906332969666
        Accuracy: 0.5184837498381458
Iteration 6000 Loss: 0.24956989288330078
        Accuracy: 0.5156674867279555
Iteration 7000 Loss: 0.24944932758808136
        Accuracy: 0.5196814709309854
Iteration 8000 Loss: 0.24956241250038147
        Accuracy: 0.5174155121066943
Iteration 9000 Loss: 0.24933822453022003
        Accuracy: 0.51670335361906
Train Loss: 0.24959170818328857
***Test Loss: 0.24946695566177368
***Test Accuracy: 0.5280331477405154
--------------------------------
  / 30 / 10 / 1 
dropout 0.50
weight_decay 1e-3
iters 10000

Thresholding
Number of Players Removed: 0
inputSize:32
Indices Generated
Training...
Iteration 0 Loss: 0.25213518738746643
        Accuracy: 0.5046614010099703
Iteration 1000 Loss: 0.24784092605113983
        Accuracy: 0.5435387802667357
Iteration 2000 Loss: 0.24775579571723938
        Accuracy: 0.5428266217791013
Iteration 3000 Loss: 0.2475428730249405
        Accuracy: 0.544153826233329
Iteration 4000 Loss: 0.2477273792028427
        Accuracy: 0.5456428848892917
Iteration 5000 Loss: 0.24763858318328857
        Accuracy: 0.5455457723682506
Iteration 6000 Loss: 0.24753746390342712
        Accuracy: 0.5439272303509
Iteration 7000 Loss: 0.24743379652500153
        Accuracy: 0.5435711511070828
Iteration 8000 Loss: 0.24780410528182983
        Accuracy: 0.5424705425352843
Iteration 9000 Loss: 0.24762140214443207
        Accuracy: 0.5418554965686909
Train Loss: 0.24756835401058197
***Test Loss: 0.24889303743839264
***Test Accuracy: 0.5347662825326945
----------------------------------------
  / 30 / 10 / 1 
dropout 0.0
weight_decay 1e-3
iters 10000
Thresholding
Number of Players Removed: 0
inputSize:32
Indices Generated
Training...
Iteration 0 Loss: 0.25071820616722107
        Accuracy: 0.49491777806551857
Iteration 1000 Loss: 0.23978659510612488
        Accuracy: 0.5892140359963745
Iteration 2000 Loss: 0.2384391874074936
        Accuracy: 0.5929690534766282
Iteration 3000 Loss: 0.23833529651165009
        Accuracy: 0.5940049203677328
Iteration 4000 Loss: 0.23827405273914337
        Accuracy: 0.594522853813285
Iteration 5000 Loss: 0.2382459193468094
        Accuracy: 0.5944904829729379
Iteration 6000 Loss: 0.23824407160282135
        Accuracy: 0.5943933704518969
Iteration 7000 Loss: 0.2382371574640274
        Accuracy: 0.5947170788553671
Iteration 8000 Loss: 0.23823514580726624
        Accuracy: 0.5940049203677328
Iteration 9000 Loss: 0.23823300004005432
        Accuracy: 0.5942962579308558
Train Loss: 0.23823225498199463
***Test Loss: 0.2532755732536316
***Test Accuracy: 0.5264793474038586
--------------------------------------
  / 30 / 10 / 1 
dropout 0.0
weight_decay 1e-2
iters 10000
convergest in two seconds
------------------------
  / 30 / 30 / 10 / 1 
  dropout 0.50
  weight_decay 1e-3
  iters 1000

  Thresholding
Number of Players Removed: 0
inputSize:32
Indices Generated
Training...
Iteration 0 Loss: 0.2539650797843933
        Accuracy: 0.49964392075618286
Iteration 1000 Loss: 0.2481720894575119
        Accuracy: 0.5421468341318141
Iteration 2000 Loss: 0.24827136099338531
        Accuracy: 0.5405282921144633
Iteration 3000 Loss: 0.2481347918510437
        Accuracy: 0.5374854331218438
Iteration 4000 Loss: 0.2480124980211258
        Accuracy: 0.5413375631231386
Iteration 5000 Loss: 0.24819883704185486
        Accuracy: 0.5398485044671759
Iteration 6000 Loss: 0.24807147681713104
        Accuracy: 0.5437977469895119
Iteration 7000 Loss: 0.24804824590682983
        Accuracy: 0.5398808753075229
Iteration 8000 Loss: 0.24808652698993683
        Accuracy: 0.5425352842159783
Iteration 9000 Loss: 0.24823400378227234
        Accuracy: 0.5424058008545902
Train Loss: 0.2478623241186142
***Test Loss: 0.2487739622592926
***Test Accuracy: 0.5359316327851871
---------------------------------------
  / 30 / 30 / 30 / 10 / 1 
  dropout 0.50
  weight_decay 1e-4
  iters 1000
Training...
Iteration 0 Loss: 0.2521301507949829
        Accuracy: 0.5005826751262463
Iteration 1000 Loss: 0.24544262886047363
        Accuracy: 0.5586883335491389
Iteration 2000 Loss: 0.24449530243873596
        Accuracy: 0.5636087012818852
Iteration 3000 Loss: 0.24374352395534515
        Accuracy: 0.5687556648970608
Iteration 4000 Loss: 0.24365046620368958
        Accuracy: 0.5700828693512884
Train Loss: 0.24316921830177307
***Test Loss: 0.2503332495689392
***Test Accuracy: 0.5288100479088437
'''


#def main():
#    #slkfdn
#    df = pd.read_csv(datapath + name + '.csv')


if __name__ == '__main__':
    training_proceedure(weight_decay = 1e-4, lr = 0.01, name = name, datapath = datapath, epochs = 3000, kfolds = 1, playerThreshhold = 50)



