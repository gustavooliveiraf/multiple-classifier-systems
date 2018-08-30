import numpy as np
import pandas as pd
import math
import random
import pdb
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
import time
from sklearn.preprocessing import LabelEncoder
t = time.time()

class Main:
    def __init__(self, data):
        self.data = data

    def KfoldNtimes(self, data, kFold, nTimes, poolSize, percentageTrainingSet):
        y = np.array(data["CLASS"])
        X = np.array(data.drop(axis=1, columns = ["CLASS"]))

        for i in range(nTimes):
            media = 0
            flag = 0
            skf = StratifiedKFold(n_splits=kFold,shuffle=True)
            # skf.get_n_splits(X, y)
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                BaggingClassifierTemp = BaggingClassifier(tree.DecisionTreeClassifier(), poolSize, percentageTrainingSet, random_state=42)

                BaggingClassifierTemp.fit(X_train, y_train)

                print(BaggingClassifierTemp.score(X_test, y_test))
                print()

#TESTE
df = pd.read_csv('./entrada.csv')
df = df.drop(axis=1, columns = ["ID"])
enc = LabelEncoder()
df.CLASS = enc.fit_transform(df.CLASS)

modelo = Main(df)
k = modelo.KfoldNtimes(df, 10, 1, 100, 1.0) # 1 time