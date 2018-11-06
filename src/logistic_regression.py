from os import listdir
from os.path import isfile, join
import time

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB


# ===============================================================================


name_to_model = {
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(),
    "Multi Layer Perceptron": MLPClassifier(alpha=1),
    "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=1000),
    "Linear SVM": SVC(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
}


# ===============================================================================


def load_batch(full_name):
    df = pd.read_csv(full_name)
    wpct = df.columns.get_loc("cum_isWin") - 1
    opp_wpct = df.columns.get_loc("opp_cum_isWin") - 1

    array = df.values[:, 1:]

    np.random.shuffle(array)

    cutoff = int(2./3 * array.shape[0])

    X_train = array[:cutoff, 1:]
    Y_train = array[:cutoff, 0]
    X_test = array[cutoff:, 1:]
    Y_test = array[cutoff:, 0]

    return X_train, Y_train, X_test, Y_test, wpct, opp_wpct


def get_naive_accuracy(x_test, y_test, wpct, opp_wpct):
    correct = 0
    total = x_test.shape[0]

    for i in range(x_test.shape[0]):
        if x_test[i, wpct] >= x_test[i, opp_wpct] and y_test[i] == 1:
            correct += 1
        if x_test[i, wpct] < x_test[i, opp_wpct] and y_test[i] == 0:
            correct += 1
    return correct/total


def batch_classify(X_train, Y_train, X_test, Y_test, wpct, opp_wpct):
    df = pd.DataFrame(data=np.zeros(shape=(len(name_to_model) + 1, 4)),
                      columns=['Name', 'Train Acc', 'Test Acc', 'Training Time (s)'])
    # Calculate naive accuracy
    naive_acc = get_naive_accuracy(X_test, Y_test, wpct, opp_wpct)
    df.loc[0, :] = ["Naive", np.nan, naive_acc, np.nan]

    # Calculate model accuracies
    name_model_pairs = list(name_to_model.items())
    for i in range(len(name_to_model)):
        name, model = name_model_pairs[i]
        t_start = time.clock()
        model.fit(X_train, Y_train)
        t_end = time.clock()

        t_diff = t_end - t_start
        train_score = model.score(X_train, Y_train)
        test_score = model.score(X_test, Y_test)

        df.loc[i + 1] = [name, train_score, test_score, t_diff]
        df.to_csv("../data/classifier_accuracies.csv")

    return df


def batch_train(model, x_train, y_train, x_test, y_test, wpct, opp_wpct):
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print("model", score)
    correct = 0
    total = x_test.shape[0]

    for i in range(x_test.shape[0]):
        if x_test[i, wpct] >= x_test[i, opp_wpct] and y_test[i] == 1:
            correct += 1
        if x_test[i, wpct] < x_test[i, opp_wpct] and y_test[i] == 0:
            correct += 1
    print("trivial", correct/total, "\n")


# =======================================================================================


if __name__ == "__main__":
    dropbox_dir = "/Users/timhenry/Dropbox (MIT)/6.867/"
    in_dir = dropbox_dir + "CUM_CONCAT"

    path = in_dir
    if not isfile(path):
        for f in listdir(path):
            team_name, ext = f.split(".")
            # ignore hidden files, etc.
            if ext.lower() != "txt" and ext.lower() != "csv":
                continue

            x_train, y_train, x_test, y_test, wpct, opp_wpct = load_batch(join(path, f))
            batch_classify(x_train, y_train, x_test, y_test, wpct, opp_wpct)
