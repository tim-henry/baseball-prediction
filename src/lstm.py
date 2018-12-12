from os import listdir, makedirs
from os.path import isdir, isfile, join
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

"""
LSTM MODEL
"""
class Network(nn.ModuleList):
    def __init__(self, input_dim, input2_dim, hidden_dim, output_dim):
        super(Network, self).__init__()
        self.input_dim = input_dim
        self.input2_dim = input2_dim
        self.hidden_dim = hidden_dim
        self.intermediate_dim = 8
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


"""
Alternate LSTM MODEL
"""
class AlternateNetwork(nn.ModuleList):
    def __init__(self, input_dim, input2_dim, hidden_dim, output_dim):
        super(Network, self).__init__()
        self.input_dim = input_dim
        self.inpu2t_dim = input2_dim
        self.hidden_dim = hidden_dim
        self.intermediate_dim = 8
        self.output_dim = output_dim

        self.batch_size = 1

        self.lstm = nn.LSTMCell(input_size=input_dim, hidden_size=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim + input2_dim, out_features=self.intermediate_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.output = nn.Linear(in_features=self.intermediate_dim, out_features=output_dim)

    """
    X: a sequence of vectors from previous game # for the teams of the previous matchup
    X2: a sequence of vectors from previous game # for the teams of this matchup
    hc: hidden state, hidden cell
    """
    def forward(self, X, X2, hc):
        sequence_size = len(X)
        output_seq = torch.empty((sequence_size, self.batch_size, self.output_dim))

        for i in range(len(X)):
            hc = self.lstm(X[i].view(1, -1).float(), hc)
            # hc = self.lstm(self.fc1(X[i].view(1, -1).float()), hc)

            hidden_state, _ = hc
            # print(hidden_state.shape, X2[i].view(1, -1).float().shape)
            # res = torch.cat([hidden_state, X2[i].float()], dim=0)
            output_seq[i] = torch.sigmoid(self.output(self.dropout(self.fc2(torch.cat((hidden_state, X2[i].view(1, -1).float()), dim=1)))))

        return output_seq.view((sequence_size, -1))

    def init_hidden(self):
        return (torch.zeros(self.batch_size, self.hidden_dim),
                torch.zeros(self.batch_size, self.hidden_dim))


# ==========================================================================================================


def clean_file(full_name, year):
    raw_data = pd.read_csv(full_name)
    n = raw_data.shape[1]

    # remove/reorganize columns
    raw_data = raw_data.drop(columns=["Name", "opp_Name", "GameNum", "opp_GameNum", "cum_GameNum", "opp_cum_GameNum", "cum_isHome",
                                      "opp_cum_isHome"])

    to_drop = []
    columns = [x for x in raw_data.columns if x.startswith("cum_") or x.startswith("batter") or x.startswith("pitcher")]
    for column in columns:
        raw_data[column] = raw_data[column] - raw_data["opp_" + column]

    for i in range((57 - 1) // 2):
        column = raw_data.columns[i + 2]
        to_drop.append(column)
        to_drop.append("opp_" + column)
        to_drop.append("opp_cum_" + column)
        to_drop.extend([y for y in raw_data.columns if y.startswith("opp_pitcher") or y.startswith("opp_batter")])#["opp_pitcher_0", "opp_pitcher_1", "opp_batter_0", "opp_batter_1"])
    raw_data = raw_data.drop(columns=to_drop)
    raw_data = raw_data.dropna(axis=0)

    return raw_data.drop(columns=["isWin"]).values, raw_data[["isWin"]].values


def clean_file_shifted_player(full_name, year):
    raw_data = pd.read_csv(full_name)
    n = raw_data.shape[1]

    # remove/reorganize columns
    if "Unnamed: 0" in raw_data.columns:
        raw_data = raw_data.drop(columns=["Unnamed: 0"])
    raw_data = raw_data.drop(columns=["Name", "Name", "opp_Name", "GameNum", "opp_GameNum", "cum_isWin",
                                      "opp_cum_isWin", "cum_GameNum", "opp_cum_GameNum", "cum_isHome",
                                      "opp_cum_isHome"])

    to_drop = []
    new_cols = [x for x in raw_data.columns if x[-1].isdigit() and not x[-2].isdigit()]
    columns = [x for x in raw_data.columns if not x.startswith("opp_") and not x.startswith("cum_")
               and x != "isWin" and x != "isHome" and not (x[-1].isdigit() and x[-2].isdigit())]

    for column in columns:
        if column in new_cols:
            continue
        raw_data["cum2_" + column] = raw_data["cum_" + column].shift(-1)

    for column in columns:
        if column in new_cols:
            continue
        raw_data["opp_cum2_" + column] = raw_data["opp_cum_" + column].shift(-1)

    for column in columns:
        if column in new_cols:
            continue
        raw_data["cum_" + column] = - raw_data["opp_cum_" + column] + raw_data["cum_" + column]
        raw_data["cum2_" + column] = - raw_data["opp_cum2_" + column] + raw_data["cum2_" + column]
        to_drop.append("opp_cum_" + column)
        to_drop.append("opp_cum2_" + column)

    for column in columns:
        raw_data[column] = - raw_data["opp_" + column] + raw_data[column]
        to_drop.append("opp_" + column)


    raw_data = raw_data.drop(columns=to_drop)

    raw_data['isWin'] = raw_data['isWin'].shift(-1)

    raw_data = raw_data.dropna(axis=0)

    return raw_data.drop(columns=["isWin"]).values, raw_data[["isWin"]].values


def clean_file_shifted_2(full_name, year):
    raw_data = pd.read_csv(full_name)
    n = raw_data.shape[1]

    # remove/reorganize columns
    raw_data = raw_data.drop(columns=["Unnamed: 0", "Name", "opp_Name", "GameNum", "opp_GameNum", "cum_isWin",
                                      "opp_cum_isWin", "cum_GameNum", "opp_cum_GameNum", "cum_isHome",
                                      "opp_cum_isHome"])

    to_drop = []
    for i in range((57 - 1) // 2):
        column = raw_data.columns[i + 2]
        raw_data["cum2_" + column] = raw_data["cum_" + column].shift(-1)

    for i in range((57 - 1) // 2):
        column = raw_data.columns[i + 2]
        raw_data["opp_cum2_" + column] = raw_data["opp_cum_" + column].shift(-1)

    X2 = None
    for i in range((57 - 1) // 2):
        column = raw_data.columns[i + 2]
        raw_data[column] = - raw_data["opp_" + column] + raw_data[column]
        raw_data["cum_" + column] = - raw_data["opp_cum_" + column] + raw_data["cum_" + column]
        X2["cum2_" + column] = - raw_data["opp_cum2_" + column] + raw_data["cum2_" + column]

        to_drop.append("opp_" + column)
        to_drop.append("cum2_" + column)
        to_drop.append("opp_cum_" + column)
        to_drop.append("opp_cum2_" + column)

    raw_data = raw_data.drop(columns=to_drop)

    raw_data['isWin'] = raw_data['isWin'].shift(-1)

    raw_data = raw_data.dropna(axis=0)

    return raw_data.drop(columns=["isWin"]).values, X2.values, raw_data[["isWin"]].values


def clean_file_shifted(full_name, year):
    raw_data = pd.read_csv(full_name)
    n = raw_data.shape[1]

    # remove/reorganize columns
    raw_data = raw_data.drop(columns=["Unnamed: 0", "Name", "opp_Name", "GameNum", "opp_GameNum", "cum_isWin",
                                      "opp_cum_isWin", "cum_GameNum", "opp_cum_GameNum", "cum_isHome",
                                      "opp_cum_isHome"])

    to_drop = []
    for i in range((57 - 1) // 2):
        column = raw_data.columns[i + 2]
        raw_data["cum2_" + column] = raw_data["cum_" + column].shift(-1)

    for i in range((57 - 1) // 2):
        column = raw_data.columns[i + 2]
        raw_data["opp_cum2_" + column] = raw_data["opp_cum_" + column].shift(-1)

    for i in range((57 - 1) // 2):
        column = raw_data.columns[i + 2]
        raw_data[column] = - raw_data["opp_" + column] + raw_data[column]
        raw_data["cum_" + column] = - raw_data["opp_cum_" + column] + raw_data["cum_" + column]
        raw_data["cum2_" + column] = - raw_data["opp_cum2_" + column] + raw_data["cum2_" + column]

        to_drop.append("opp_" + column)
        to_drop.append("opp_cum_" + column)
        to_drop.append("opp_cum2_" + column)

    raw_data = raw_data.drop(columns=to_drop)

    raw_data['isWin'] = raw_data['isWin'].shift(-1)

    raw_data = raw_data.dropna(axis=0)

    return raw_data.drop(columns=["isWin"]).values, raw_data[["isWin"]].values


def clean_file_shifted_team(full_name, year):
    raw_data = pd.read_csv(full_name)
    team_data = pd.read_csv(dropbox_dir + "teams-one-hot.csv", index_col=0)

    n = raw_data.shape[1]

    team_features = pd.DataFrame(index=raw_data.index, columns=team_data.columns)
    opp_features = pd.DataFrame(index=raw_data.index, columns=team_data.columns)

    for idx in range(raw_data.shape[0]):
        team_features.at[idx] = team_data.loc[raw_data.at[idx, "Name"]]
        opp_features.at[idx] = team_data.loc[raw_data.at[idx, "opp_Name"]]
    raw_data = raw_data.join(team_features.astype('float64').sub(opp_features.astype('float64')))

    # remove/reorganize columns
    if "Unnamed: 0" in raw_data.columns:
        raw_data = raw_data.drop(columns=["Unnamed: 0"])
    raw_data = raw_data.drop(columns=["Name", "opp_Name", "GameNum", "opp_GameNum", "cum_isWin",
                                      "opp_cum_isWin", "cum_GameNum", "opp_cum_GameNum", "cum_isHome",
                                      "opp_cum_isHome"])

    to_drop = []
    for i in range((57 - 1) // 2):
        column = raw_data.columns[i + 2]
        raw_data["cum2_" + column] = raw_data["cum_" + column].shift(-1)

    for i in range((57 - 1) // 2):
        column = raw_data.columns[i + 2]
        raw_data["opp_cum2_" + column] = raw_data["opp_cum_" + column].shift(-1)

    for i in range((57 - 1) // 2):
        column = raw_data.columns[i + 2]
        raw_data[column] = - raw_data["opp_" + column] + raw_data[column]
        raw_data["cum_" + column] = - raw_data["opp_cum_" + column] + raw_data["cum_" + column]
        raw_data["cum2_" + column] = - raw_data["opp_cum2_" + column] + raw_data["cum2_" + column]

        to_drop.append("opp_" + column)
        to_drop.append("opp_cum_" + column)
        to_drop.append("opp_cum2_" + column)

    raw_data = raw_data.drop(columns=to_drop)

    raw_data['isWin'] = raw_data['isWin'].shift(-1)
    raw_data['year'] = int(year[2:]) - 1975

    raw_data = raw_data.dropna(axis=0)
    return raw_data.drop(columns=["isWin"]).values, raw_data[["isWin"]].values


# Clean Data w/ team vectors
def clean_file_team(full_name, year):
    raw_data = pd.read_csv(full_name)
    team_data = pd.read_csv(dropbox_dir + "teams-one-hot.csv", index_col=0)

    n = raw_data.shape[1]

    team_features = pd.DataFrame(index=raw_data.index, columns=team_data.columns)
    opp_features = pd.DataFrame(index=raw_data.index, columns=team_data.columns)

    for idx in range(raw_data.shape[0]):
        team_features.at[idx] = team_data.loc[raw_data.at[idx, "Name"]]
        opp_features.at[idx] = team_data.loc[raw_data.at[idx, "opp_Name"]]
    raw_data = raw_data.join(team_features.astype('float64').sub(opp_features.astype('float64')))

    # remove/reorganize columns
    raw_data = raw_data.drop(columns=["Unnamed: 0", "Name", "opp_Name", "GameNum", "opp_GameNum", "cum_isWin",
                                      "opp_cum_isWin", "cum_GameNum", "opp_cum_GameNum", "cum_isHome",
                                      "opp_cum_isHome"])

    to_drop = []
    for i in range((57 - 1) // 2):
        column = raw_data.columns[i + 2]
        to_drop.append(column)
        to_drop.append("opp_" + column)
        to_drop.append("cum_" + column)
        to_drop.append("opp_cum_" + column)
    raw_data = raw_data.drop(columns=to_drop)
    raw_data = raw_data.dropna(axis=0)

    return raw_data.drop(columns=["isWin"]).values, raw_data[["isWin"]].values

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
            seq, label = clean_file_shifted_player(full_name, year)
            if np.isnan(seq).any() or np.isnan(label).any():
                print("WARNING: nan's found: ", year, team_name)
                continue
            seqs.append(seq)
            labels.append(label)
    return seqs, labels

# Load Data
def get_data_alternate(in_dir, year):
    seqs1 = []
    seqs2 = []
    labels = []
    path = join(in_dir, year)
    if not isfile(path):
        for f in listdir(path):
            full_name = join(in_dir, year, f)
            team_name, ext = f.split(".")
            # ignore hidden files, etc.
            if ext.lower() != "txt" and ext.lower() != "csv":
                continue
            seq1, seq2, label = clean_file_shifted2(full_name, year)
            if np.isnan(seq1).any() or np.isnan(label).any():
                # print(year, team_name)
                print("WARNING: nan's found: ", year, team_name)
                continue
            seqs1.append(seq1)
            seqs2.append(seq2)
            labels.append(label)
    return seqs1, seqs2, labels


# Evaluate
def get_acc(y_pred, y, burn_index=0):
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
    dropbox_dir = "/Users/timhenry/Dropbox (MIT)/6 (1).867/"
    # in_dir = dropbox_dir + "data_clean_csv_wins_cumulated_MA"
    in_dir = dropbox_dir + "data_clean_csv_wins_cumulated_withplayers/"
    # in_dir = dropbox_dir + "data_clean_csv_wins_cumulated_withplayers_transformed/mixture/num_degrees5"
    train_seasons = [x for x in range(2010, 2015)]
    test_seasons = [x for x in range(2016, 2018)]
    # input_dim = 95
    input_dim = 3665
    input_dim2 = 0
    # Hyper-params
    hidden_dim = 16
    output_dim = 1  # classification
    model = Network(input_dim=input_dim, input2_dim=input_dim2, hidden_dim=hidden_dim, output_dim=output_dim)
    model = model.to(device)

    criterion = nn.BCELoss()
    lr = 0.03
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

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
                X1, Y = torch.from_numpy(train_X[seq_idx]), torch.from_numpy(train_Y[seq_idx])
                X1 = X1.to(device)
                Y = Y.to(device)

                optimizer.zero_grad()
                Y_pred = model(X1, model.init_hidden())
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
    for season in test_seasons:
        test_X1, test_Y = get_data(in_dir, "GL" + str(season))

        season_correct = 0
        season_total = 0
        for seq_idx in range(len(test_X1)):
            X1, Y = torch.from_numpy(test_X1[seq_idx]), torch.from_numpy(
                test_Y[seq_idx])
            X1 = X1.to(device)
            Y = Y.to(device)

            Y_pred = model(X1, model.init_hidden())

            num_correct, num_total = get_acc(Y_pred, Y)
            # print("Acc for season {0}, team {1}: {2:3.3f}".format(season, seq_idx, num_correct / num_total))
            season_correct += num_correct
            season_total += num_total

        print("Season {0} accuracy: {1:3.3f}".format(season, season_correct / season_total))
        total_correct += season_correct
        total += season_total

    print("Overall accuracy: {0:3.3f}".format(total_correct / total))
