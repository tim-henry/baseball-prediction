from os import listdir, makedirs
from os.path import isdir, isfile, join
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

"""
LSTM MODEL

input_dim: size of stat vector
hidden_dim: TODO
output_dim: 1 for classification
"""
class Network(nn.ModuleList):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Network, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.batch_size = 1

        self.lstm = nn.LSTMCell(input_size=input_dim, hidden_size=hidden_dim)

        self.output = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    """
    X: a sequence of vectors from previous game # for the teams of this matchup
    hc: hidden state, hidden cell
    """
    def forward(self, X, hc):
        sequence_size = len(X)
        output_seq = torch.empty((sequence_size, self.batch_size, self.output_dim))

        for i in range(len(X)):
            hc = self.lstm(X[i].view(1, -1).float(), hc)

            hidden_state, _ = hc

            output_seq[i] = torch.sigmoid(self.output(hidden_state))

        return output_seq.view((sequence_size, -1))

    def init_hidden(self):
        return (torch.zeros(self.batch_size, self.hidden_dim),
                torch.zeros(self.batch_size, self.hidden_dim))


# Clean Data
def clean_file(full_name):
    raw_data = pd.read_csv(full_name)
    n = raw_data.shape[1]

    raw_data = raw_data.drop(raw_data.index[0])
    raw_data = raw_data.drop(raw_data.index[0])
    labels = raw_data[["isWin"]]

    # remove/reorganize columns
    raw_data = raw_data.drop(columns=["Unnamed: 0", "isWin", "Name", "opp_Name", "GameNum", "opp_GameNum",
                                      "cum_isWin", "opp_cum_isWin", "cum_GameNum", "opp_cum_GameNum", "cum_isHome", "opp_cum_isHome"])

    to_drop = []
    for i in range((57 - 1) // 2):
        column = raw_data.columns[i + 1]
        raw_data[column] = raw_data[column] - raw_data["opp_" + column]
        raw_data["cum_" + column] = raw_data["cum_" + column] - raw_data["opp_cum_" + column]
        to_drop.append("opp_" + column)
        to_drop.append("opp_cum_" + column)
    raw_data = raw_data.drop(columns=to_drop)

    return raw_data.values, labels.values

# Load Data
def get_data(in_dir, year):
    seqs = []
    labels = []
    path = join(in_dir, year)
    if not isfile(path):
        for f in listdir(path):
            full_name = join(in_dir, year, f)
            team_name, ext = f.split(".")
            # ignore hidden files, etc.
            if ext.lower() != "txt" and ext.lower() != "csv":
                continue
            seq, label = clean_file(full_name)
            if np.isnan(seq).any() or np.isnan(label).any():
                print("WARNING: nan's found")
                continue
            seqs.append(seq)
            labels.append(label)
    return seqs, labels


# Evaluate
def get_acc(y_pred, y, burn_index=10):
    tot = y_pred[burn_index:].shape[0]
    y_pred_rnd = torch.round(y_pred).long()
    correct = torch.sum(y_pred_rnd[burn_index:] == y.long()[burn_index:])
    return correct.item(), tot


# ==========================================================================================================


if __name__ == "__main__":
    # Model output
    output_dir = "../cdmodels/"
    model_name = "model"
    if not isdir(output_dir):
        makedirs(output_dir)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Data
    dropbox_dir = "/Users/timhenry/Dropbox (MIT)/6.867/"
    in_dir = dropbox_dir + "data_clean_csv_wins_cumulated"
    train_seasons = [x for x in range(2008, 2013)]
    test_seasons = [x for x in range(2014, 2017)]
    input_dim = 57

    # Hyper-params
    hidden_dim = 9  # TODO ?
    output_dim = 1  # classification
    model = Network(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    model = model.to(device)

    criterion = nn.BCELoss()  # TODO custom loss function? (burn first n predictions?)
    lr = 0.01
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    print("============== Training ==============")
    model.train()

    epochs = 10
    for epoch in range(epochs):
        total_correct = 0
        total = 0
        print("--- Epoch " + str(epoch) + " ---")
        for season in train_seasons:
            train_X, train_Y = get_data(in_dir, "GL" + str(season))

            season_correct = 0
            season_total = 0
            for seq_idx in range(len(train_X)):
                X, Y = torch.from_numpy(train_X[seq_idx][:-1]), torch.from_numpy(train_Y[seq_idx][1:])
                X = X.to(device)
                Y = Y.to(device)

                optimizer.zero_grad()
                Y_pred = model(X, model.init_hidden())
                loss = criterion(Y_pred, Y.long().float())
                loss.backward()

                optimizer.step()
                num_correct, num_total = get_acc(Y_pred, Y)
                # print("Acc for season {0}, team {1}: {2:3.3f}".format(season, seq_idx, num_correct / num_total))
                season_correct += num_correct
                season_total += num_total

            # print("Season {0} accuracy: {1:3.3f}".format(season, season_correct / season_total))
            total_correct += season_correct
            total += season_total

        print("Overall epoch {0} accuracy: {1:3.3f}".format(epoch, total_correct / total))
        torch.save(model.state_dict(), output_dir + model_name)

    # Eval
    print("============== Evaluating ==============")
    model.eval()
    total_correct = 0
    total = 0
    for season in train_seasons:#test_seasons: TODO
        test_X, test_Y = get_data(in_dir, "GL" + str(season))

        season_correct = 0
        season_total = 0
        for seq_idx in range(len(test_X)):
            X, Y = torch.from_numpy(test_X[seq_idx]), torch.from_numpy(test_Y[seq_idx])
            X = X.to(device)
            Y = Y.to(device)

            Y_pred = model(X, model.init_hidden())

            num_correct, num_total = get_acc(Y_pred, Y)
            print("Acc for season {0}, team {1}: {2:3.3f}".format(season, seq_idx, num_correct / num_total))
            season_correct += num_correct
            season_total += num_total

        print("Season {0} accuracy: {1:3.3f}".format(season, season_correct / season_total))
        total_correct += season_correct
        total += season_total

    print("Overall accuracy: {0:3.3f}".format(total_correct / total))
