from sklearn.linear_model import LogisticRegression
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np


def trim_to_cum(df):
    col_names = df.columns.values

    for k, name in enumerate(col_names):
        if (not ('isWin' in name)) and (not('cum' in name) and (not ('isHome' in name))):
            df = df.drop(col_names[k], axis=1)
    return df


def load_batch(full_name):
    df = trim_to_cum(pd.read_csv(full_name))
    wpct = df.columns.get_loc("cum_isWin")
    opp_wpct = df.columns.get_loc("opp_cum_isWin")

    array = df.values[1:, :]

    np.random.shuffle(array)

    cutoff = int(2./3 * array.shape[0])

    X_train = array[:cutoff, 1:]
    Y_train = array[:cutoff, 0]
    X_test = array[cutoff:, 1:]
    Y_test = array[cutoff:, 0]

    return X_train, Y_train, X_test, Y_test, wpct, opp_wpct


def batch_train(model, x_train, y_train, x_test, y_test, wpct, opp_wpct):
    model.partial_fit(x_train, y_train, classes=[0, 1])
    score = model.score(x_test, y_test)
    print("model", score)
    correct = 0
    total = x_test.shape[0]

    for i in range(x_test.shape[0]):
        if x_test[i, wpct] == x_test[i, opp_wpct]:
            correct += 0.5
        if x_test[i, wpct] > x_test[i, opp_wpct] and y_test[i] == 1:
            correct += 1
        if x_test[i, wpct] < x_test[i, opp_wpct] and y_test[i] == 0:
            correct += 1
    print("trivial", correct/total, "\n")


model = LogisticRegression()

dropbox_dir = "/Users/timhenry/Dropbox (MIT)/6.867/"
in_dir = dropbox_dir + "CUM"

X_train, Y_train, X_test, Y_test = None, None, None, None
path = in_dir
if not isfile(path):
    for f in listdir(path):
        team_name, ext = f.split(".")
        # ignore hidden files, etc.
        if ext.lower() != "txt" and ext.lower() != "csv":
            continue
        x_train, y_train, x_test, y_test, wpct, opp_wpct = load_batch(join(path, f))
        X_train = x_train if X_train is None else np.vstack([X_train, x_train])
        Y_train = y_train if Y_train is None else np.concatenate([Y_train, y_train])
        X_test = x_test if X_test is None else np.vstack([X_test, x_test])
        Y_test = y_test if Y_test is None else np.concatenate([Y_test, y_test])

model.fit(X_train, Y_train)
score = model.score(X_test, Y_test)
print("Model:", score)

correct = 0
total = X_test.shape[0]

for i in range(X_test.shape[0]):
    if X_test[i, wpct] == X_test[i, opp_wpct]:
        correct += 0.5
    if X_test[i, wpct] > X_test[i, opp_wpct] and Y_test[i] == 1:
        correct += 1
    if X_test[i, wpct] < X_test[i, opp_wpct] and Y_test[i] == 0:
        correct += 1
print("trivial", correct / total, "\n")

