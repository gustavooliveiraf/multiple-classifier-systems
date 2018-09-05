import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import itertools
import seaborn as sns
from collections import Counter
from scipy.stats import  wilcoxon

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE 

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
                X_train, X_test = x[train_index], x[test_index]
                Y_train, Y_test = y[train_index], y[test_index]

                x_train, y_train = SMOTE().fit_sample(X_train, Y_train)
                x_test, y_test = SMOTE().fit_sample(X_test, Y_test)
                # print(format(Counter(y)))   
                
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

            decisionTree[0].append(np.mean(decisionTreeTemp[0]))
            decisionTree[1].append(np.mean(decisionTreeTemp[1]))
            decisionTree[2].append(np.mean(decisionTreeTemp[2]))
            decisionTree[3].append(np.mean(decisionTreeTemp[3]))

            perceptron[0].append(np.mean(perceptronTemp[0]))
            perceptron[1].append(np.mean(perceptronTemp[1]))
            perceptron[2].append(np.mean(perceptronTemp[2]))
            perceptron[3].append(np.mean(perceptronTemp[3]))

        return (decisionTree, perceptron)

    def plot(self, bagging, random_subspace, flag):
        barWidth = 0.1
        bars = []
        
        if (flag == 0):
            for i in range(self.metrics_length): # bagging decisionTree
                bars.append(list(itertools.chain(*bagging[0][i])))
        if (flag == 1):
            for i in range(self.metrics_length): # bagging perceptron
                bars.append(list(itertools.chain(*bagging[1][i])))
        if (flag == 2):
            for i in range(self.metrics_length): # random_subspace decisionTree
                bars.append(list(itertools.chain(*random_subspace[0][i])))
        if (flag == 3):
            for i in range(self.metrics_length): # random_subspace perceptron
                bars.append(list(itertools.chain(*random_subspace[1][i])))        
        # print(bars)
        r = [np.arange((len(bars[0]))) for x in range(self.metrics_length)]
        for i in range(1, self.metrics_length):
            r[i] = [x + barWidth for x in r[i-1]]

        sns.set()
        rcParams['figure.figsize'] = 8,3

        color = ['blue', 'red', 'green', 'cyan']
        label = ['t-acerto', 'AUC', 'g-mean', 'f-measure']
        for i in range(self.metrics_length):
            plt.bar(r[i], bars[i], width = barWidth, color = color[i], edgecolor = 'black', label=label[i])

        plt.xticks([r + barWidth + 0.05 for r in range(len(bars[0])+2)], ['50%', '60%', '70%', '80%', '90%', '100%', '', ''])
        plt.xlabel('Porcentagem do conjunto de treinamento')
        plt.ylabel('Escore')
        plt.legend()
        plt.ylim([0, 1.19])
        plt.show()

    def wilcoxon_test(self, x, y):
        p_value = wilcoxon(x, y)
        # print(p_value)

        return p_value

def test(self):
    max_samples = 0.5
    max_features = 1
    bagging = ([[] for x in range(self.metrics_length)], [[] for x in range(self.metrics_length)])
    random_subspace = ([[] for x in range(self.metrics_length)], [[] for x in range(self.metrics_length)])

    for _ in range(self.train_length):
        print("=========== " + str(round(max_samples * 100)) + "% ===========")
        bagging_temp = modelo.calc_metrics(10, 1, 100, max_samples, 1.0, True, True)
        random_subspace_temp = modelo.calc_metrics(10, 1, 100, max_samples, max_features/2, True, True)
        for i in range(self.classifiers_length):
            for j in range(self.metrics_length):
                bagging[i][j].append(bagging_temp[i][j])
                random_subspace[i][j].append(random_subspace_temp[i][j])      
        max_samples = round(max_samples + 0.1, 1)
    print("=========== plotando ===========")
    self.plot(bagging, random_subspace, 0)
    self.plot(bagging, random_subspace, 1)
    self.plot(bagging, random_subspace, 2)
    self.plot(bagging, random_subspace, 3)



# test
# df = df.drop(axis=1, columns = ["ID"])
# df = pd.read_csv('./entrada.csv')
# df = pd.read_csv('./cm1.csv')
# df = pd.read_csv('./jm1.csv')
df = pd.read_csv('./kc2.csv')

enc = LabelEncoder()
df.CLASS = enc.fit_transform(df.CLASS)

modelo = Main(df)
# modelo.wilcoxon_test(x,y)

test(modelo)