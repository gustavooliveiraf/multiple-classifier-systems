import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import seaborn as max_samples
sns.set()

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from sklearn import tree
from sklearn import linear_model

from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import f1_score
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score

import pdb
class Main:
    def __init__(self, data):
        self.data = data
        self.metrics_length = 4
        self.classifiers_length = 2
        self.train_length = 6

    def calc_metrics(self, k_fold, n_times, pool_size, max_samples, max_features, bootstrap, bootstrap_features):
        y = np.array(self.data["CLASS"])
        x = np.array(self.data.drop(axis=1, columns = ["CLASS"]))

        decisionTree = [[] for x in range(self.metrics_length)]
        perceptron = [[] for x in range(self.metrics_length)]

        for i in range(n_times):
            skf = StratifiedKFold(n_splits=k_fold,shuffle=True)
            decisionTreeTemp = [[] for x in range(self.metrics_length)]
            perceptronTemp = [[] for x in range(self.metrics_length)]

            for train_index, test_index in skf.split(x, y):
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]

                
                BaggingClassifierDecisionTree = BaggingClassifier(tree.DecisionTreeClassifier(), pool_size, max_samples, max_features, bootstrap, bootstrap_features)
                BaggingClassifierDecisionTree.fit(x_train, y_train)
                decisionTreeTemp[0].append(BaggingClassifierDecisionTree.score(x_test, y_test))
                decisionTreeTemp[1].append(roc_auc_score(BaggingClassifierDecisionTree.predict(x_test), y_test))
                decisionTreeTemp[2].append(geometric_mean_score(BaggingClassifierDecisionTree.predict(x_test), y_test))
                decisionTreeTemp[3].append(f1_score(BaggingClassifierDecisionTree.predict(x_test), y_test))

                
                BaggingClassifierPerceptron = BaggingClassifier(linear_model.Perceptron(max_iter=5), pool_size, max_samples, max_features, bootstrap, bootstrap_features)
                BaggingClassifierPerceptron.fit(x_train, y_train)
                perceptronTemp[0].append(BaggingClassifierPerceptron.score(x_test, y_test))
                perceptronTemp[1].append(roc_auc_score(BaggingClassifierPerceptron.predict(x_test), y_test))
                perceptronTemp[2].append(geometric_mean_score(BaggingClassifierPerceptron.predict(x_test), y_test))
                perceptronTemp[3].append(f1_score(BaggingClassifierPerceptron.predict(x_test), y_test))

                # print("---------------------- fold " + str(i) + " --------------------------------")

            decisionTree[0].append(np.mean(decisionTreeTemp[0]))
            decisionTree[1].append(np.mean(decisionTreeTemp[1]))
            decisionTree[2].append(np.mean(decisionTreeTemp[2]))
            decisionTree[3].append(np.mean(decisionTreeTemp[3]))

            perceptron[0].append(np.mean(perceptronTemp[0]))
            perceptron[1].append(np.mean(perceptronTemp[1]))
            perceptron[2].append(np.mean(perceptronTemp[2]))
            perceptron[3].append(np.mean(perceptronTemp[3]))

        return (decisionTree, perceptron)

    def plot(self, bagging, random_subspace):
        barWidth = 0.1
        bars = []
        for i in range(self.metrics_length):
            bars.append(list(itertools.chain(*bagging[0][i])))
        
        r = [np.arange((len(bars[0]))) for x in range(self.train_length)]
        for i in range(1, self.train_length):
            r[i] = [x + barWidth for x in r[i-1]]

        for i in range(self.metrics_length):
            plt.bar(r[i], bars[i], width = barWidth, color = 'blue', edgecolor = 'black', label='poacee')

        plt.ylabel('accuracy')
        plt.legend()
        plt.show()

def test(self):
    max_samples = 0.5
    max_features = 1
    bagging = ([[] for x in range(self.metrics_length)], [[] for x in range(self.metrics_length)])
    random_subspace = ([[] for x in range(self.metrics_length)], [[] for x in range(self.metrics_length)])


    for _ in range(self.train_length):
        print("=========== para " + str(max_samples * 100) + "% ===========")

        bagging_temp = modelo.calc_metrics(10, 1, 100, max_samples, 1.0, True, True)
        random_subspace_temp = modelo.calc_metrics(10, 1, 100, max_samples, max_features/2, True, True)
        for i in range(self.classifiers_length):
            for j in range(self.metrics_length):
                bagging[i][j].append(bagging_temp[i][j])
                random_subspace[i][j].append(random_subspace_temp[i][j])      
        max_samples += 0.1

    self.plot(bagging, random_subspace)




# TESTE
# df = df.drop(axis=1, columns = ["ID"])
# df = pd.read_csv('./entrada.csv')
# df = pd.read_csv('./cm1.csv')
# df = pd.read_csv('./jm1.csv')
df = pd.read_csv('./kc2.csv')

enc = LabelEncoder()
df.CLASS = enc.fit_transform(df.CLASS)

modelo = Main(df)
test(modelo)