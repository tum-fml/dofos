# ######################################################################################################################
#
# Definition einiger Klassen um die Übersichtlichkeit zu wahren.
#
# ######################################################################################################################
import glob
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
# from sklearn.preprocessing import MinMaxScaler

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score



import joblib


class data:  # Get pattern of each operationg condition
    def __init__(self, path, label, operating_condition):
        self.path = path
        self.label = label
        self.operating_condition = operating_condition

    def list(self, dataframe_list, labels):
        txt_files = glob.glob(self.path + "**/*.txt", recursive=True)
        label_list = np.zeros(len(txt_files))
        label_list.fill(self.label)
        df_list = [pd.read_csv(txt, sep=' ', names=['x', 'y', 'z'],  header=None) for txt in txt_files]

        return dataframe_list.extend(df_list), labels.extend(label_list)  # extend is iterable


class features:  #
    def __init__(self, features, feature_matrix, b):  # n for main or training skript
        self.features = features
        self.feature_matrix = feature_matrix
        self.b = b
        # if features > 9:
        #     print("Choose a Value for Features from 1 to 9. There are not more than 9 Features available")

    def peak_to_peak(self, df, t, runVariable):

        # if self.b == 2:
        #     self.feature_matrix[t, 0 + runVariable] = abs(max(df['x']) - min(df['x']))
        #     self.feature_matrix[t, 1 + runVariable] = abs(max(df['y']) - min(df['y']))
        #     self.feature_matrix[t, 2 + runVariable] = abs(max(df['z']) - min(df['z']))
        # else:
        #     self.feature_matrix[0 + runVariable] = abs(max(df['x']) - min(df['x']))
        #     self.feature_matrix[1 + runVariable] = abs(max(df['y']) - min(df['y']))
        #     self.feature_matrix[2 + runVariable] = abs(max(df['z']) - min(df['z']))
        #
        # return  self.feature_matrix

        if self.b == 2:
            self.feature_matrix[t, 0 + runVariable] = max(df['x']) - min(df['x'])
            self.feature_matrix[t, 1 + runVariable] = max(df['y']) - min(df['y'])
            self.feature_matrix[t, 2 + runVariable] = max(df['z']) - min(df['z'])
        else:
            self.feature_matrix[0 + runVariable] = max(df['x']) - min(df['x'])
            self.feature_matrix[1 + runVariable] = max(df['y']) - min(df['y'])
            self.feature_matrix[2 + runVariable] = max(df['z']) - min(df['z'])

        return  self.feature_matrix

    def min(self, df, t, runVariable):

        if self.b == 2:
            self.feature_matrix[t, 0 + runVariable] = min(df['x'])
            self.feature_matrix[t, 1 + runVariable] = min(df['y'])
            self.feature_matrix[t, 2 + runVariable] = min(df['z'])
        else:
            self.feature_matrix[0 + runVariable] = min(df['x'])
            self.feature_matrix[1 + runVariable] = min(df['y'])
            self.feature_matrix[2 + runVariable] = min(df['z'])

        return self.feature_matrix

    def max(self, df, t, runVariable):

        if self.b == 2:
            self.feature_matrix[t, 0 + runVariable] = max(df['x'])
            self.feature_matrix[t, 1 + runVariable] = max(df['y'])
            self.feature_matrix[t, 2 + runVariable] = max(df['z'])
        else:
            self.feature_matrix[0 + runVariable] = max(df['x'])
            self.feature_matrix[1 + runVariable] = max(df['y'])
            self.feature_matrix[2 + runVariable] = max(df['z'])

        return self.feature_matrix

    def euclidean_Dist(self, df, t, runVariable):
        pattern_length = len(df)
        xy_reference = np.zeros(pattern_length)

        if self.b == 2:
            self.feature_matrix[t, 0 + runVariable] = np.linalg.norm(df['x'] - xy_reference)
            self.feature_matrix[t, 1 + runVariable] = np.linalg.norm(df['y'] - xy_reference)
            self.feature_matrix[t, 2 + runVariable] = np.linalg.norm(df['z'] - xy_reference)
        else:
            self.feature_matrix[0 + runVariable] = np.linalg.norm(df['x'] - xy_reference)
            self.feature_matrix[1 + runVariable] = np.linalg.norm(df['y'] - xy_reference)
            self.feature_matrix[2 + runVariable] = np.linalg.norm(df['z'] - xy_reference)

        return self.feature_matrix

    def fft(self, df, t, runVariable):

        if self.b == 2:
            self.feature_matrix[t, 0 + runVariable] = np.abs(sum(np.fft.fft(df['x'])))
            self.feature_matrix[t, 1 + runVariable] = np.abs(sum(np.fft.fft(df['y'])))
            self.feature_matrix[t, 2 + runVariable] = np.abs(sum(np.fft.fft(df['z'])))
        else:
            self.feature_matrix[0 + runVariable] = np.abs(sum(np.fft.fft(df['x'])))
            self.feature_matrix[1 + runVariable] = np.abs(sum(np.fft.fft(df['y'])))
            self.feature_matrix[2 + runVariable] = np.abs(sum(np.fft.fft(df['z'])))

        return self.feature_matrix

    def var(self, df, t, runVariable):

        if self.b == 2:
            self.feature_matrix[t, 0 + runVariable] = np.var(df['x'])
            self.feature_matrix[t, 1 + runVariable] = np.var(df['y'])
            self.feature_matrix[t, 2 + runVariable] = np.var(df['z'])
        else:
            self.feature_matrix[0 + runVariable] = np.var(df['x'])
            self.feature_matrix[1 + runVariable] = np.var(df['y'])
            self.feature_matrix[2 + runVariable] = np.var(df['z'])

        return self.feature_matrix

    def std(self, df, t, runVariable):

        if self.b == 2:
            self.feature_matrix[t, 0 + runVariable] = np.std(df['x'])
            self.feature_matrix[t, 1 + runVariable] = np.std(df['y'])
            self.feature_matrix[t, 2 + runVariable] = np.std(df['z'])
        else:
            self.feature_matrix[0 + runVariable] = np.std(df['x'])
            self.feature_matrix[1 + runVariable] = np.std(df['y'])
            self.feature_matrix[2 + runVariable] = np.std(df['z'])
        return self.feature_matrix

    def skewness(self, df, t, runVariable):

        if self.b == 2:
            self.feature_matrix[t, 0 + runVariable] = df['x'].skew()
            # print(self.feature_matrix[t, 0 + runVariable])
            self.feature_matrix[t, 1 + runVariable] = df['y'].skew()
            self.feature_matrix[t, 2 + runVariable] = df['z'].skew()
        else:
            self.feature_matrix[0 + runVariable] = df['x'].skew()
            self.feature_matrix[1 + runVariable] = df['y'].skew()
            self.feature_matrix[2 + runVariable] = df['z'].skew()
        return self.feature_matrix

    def kurtosis(self, df, t, runVariable):

        if self.b == 2:
            self.feature_matrix[t, 0 + runVariable] = df['x'].kurt()
            # print(self.feature_matrix[t, 0 + runVariable])
            self.feature_matrix[t, 1 + runVariable] = df['y'].kurt()
            self.feature_matrix[t, 2 + runVariable] = df['z'].kurt()
        else:
            self.feature_matrix[0 + runVariable] = df['x'].kurt()
            self.feature_matrix[1 + runVariable] = df['y'].kurt()
            self.feature_matrix[2 + runVariable] = df['z'].kurt()
        return self.feature_matrix

class training:  # All models have multi label input and single label output
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.features_train, self.features_test, self.labels_train, self.labels_test = \
            train_test_split(self.features, self.labels, test_size=0.3, random_state=30, shuffle=True)

        print("Set up Training and Test data")
        mylabels = list(set(self.labels))
        self.mylabels = mylabels
        print(mylabels)
        print()

    # scikit
    def knn(self, n):  # k-Nearest-Neighbours
        neigh = KNeighborsClassifier(n_neighbors=n)
        neigh.fit(self.features_train, self.labels_train)
        labels_pred = neigh.predict(self.features_test)

        scores = cross_val_score(neigh, self.features_train, self.labels_train, cv=10)
        print("knn Cross-validation")
        print(scores)
        print("Mean: ", np.mean(scores))

        name = self.knn.__name__
        self.accuracy_score(name, labels_pred)

        return neigh

    def svm(self):  # Support Vector Machine
        svm = SVC(kernel='rbf', C=1.0, gamma=0.5)
        svm.fit(self.features_train, self.labels_train)
        labels_pred = svm.predict(self.features_test)

        scores = cross_val_score(svm, self.features_train, self.labels_train, cv=10)
        print("SVM Cross-validation")
        print(scores)
        print("Mean: ", np.mean(scores))

        name = self.svm.__name__
        self.accuracy_score(name, labels_pred)

        return svm

    def dt(self):  # Decision Tree
        dt = tree.DecisionTreeClassifier(max_features=2, random_state=0)
        dt.fit(self.features_train, self.labels_train)
        labels_pred = dt.predict(self.features_test)

        scores = cross_val_score(dt, self.features_train, self.labels_train, cv=10)
        print("DT Cross-validation")
        print(scores)
        print("Mean: ", np.mean(scores))

        name = self.dt.__name__
        self.accuracy_score(name, labels_pred)

        return dt

    def rf(self):  # Random Forest
        rf = RandomForestClassifier(n_estimators=20, random_state=0)
        rf.fit(self.features_train, self.labels_train)
        labels_pred = rf.predict(self.features_test)

        scores = cross_val_score(rf, self.features_train, self.labels_train, cv=10)
        print("RF Cross-validation")
        print(scores)
        print("Mean: ", 100*np.mean(scores))

        name = self.rf.__name__
        self.accuracy_score(name, labels_pred)

        return rf

    def accuracy_score(self, name, labels_pred):

        print("-------------------------------------------------------------------------------------------------------")
        print("%s" % name)
        print()
        print("Accuracy:", 100*metrics.accuracy_score(self.labels_test, labels_pred))
        print()

        # cm = metrics.confusion_matrix(labels_pred, self.labels_test, labels=self.mylabels)
        # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # # print(cm_normalized)
        #
        # fig, ax = plt.subplots(figsize=(5, 4))
        # sns.heatmap(cm_normalized, annot=True, fmt=".2f",
        #             cmap='Blues', square=True,xticklabels=[1,2,3,4,5,6,7], yticklabels=[1,2,3,4,5,6,7])
        # ax.set_xlabel('Predicted Activity')
        # ax.set_ylabel('True Activity', )
        # plt.tight_layout()
        # plt.show()

        # print("FScore:",  100*metrics.f1_score(self.labels_test, labels_pred, average='micro'))
        # print()

        # precision, recall, fscore, support = metrics.precision_recall_fscore_support(self.labels_test, labels_pred)
        #
        # print('precision: {}'.format(precision))
        # print('recall: {}'.format(recall))
        # print('fscore: {}'.format(fscore))
        # print('support: {}'.format(support))

        # # 7 Features
        # print('proposed & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\' % (
        # 100.0 * cm_normalized[0, 0], 100.0 * cm_normalized[1, 1], 100.0 * cm_normalized[2, 2],
        # 100.0 * cm_normalized[3, 3], 100.0 * cm_normalized[4, 4], 100.0 * cm_normalized[5, 5],
        # 100.0 * cm_normalized[6, 6]))

        # # 3 Features
        # print('proposed & %.2f & %.2f & %.2f \\\\' % (
        # 100.0 * cm_normalized[0, 0], 100.0 * cm_normalized[1, 1], 100.0 * cm_normalized[2, 2]))

        print("-------------------------------------------------------------------------------------------------------")


class save_model:
    def __init__(self, *args):
        self.args = args
        print(self.args)

    def save(self):
        for arg in self.args:
            joblib.dump(arg, "%s_%sconditions.pkl" % (str(arg)[0], 3))

# Coming soon ...
class plot:
    def __init__(self):
        print("Hallo")

    def plot_features(self):
        print("Hallo")
        # Ermittelte feature plotten 3-Achs

    def plot_accuracy(self):
        print("Hallo")
        # Plot der Confusion-Matrix



def med_filter(df, window):
    df = df.rolling(window).median()
    return df

def clean_data(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

def min_max_scaler(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    df[["x_scaled", "y_scaled", "z_scaled"]] = min_max_scaler.fit_transform(df[["x", "y", "z"]])
    return df

def normalize(df):
    df[["x_scaled", "y_scaled", "z_scaled"]] = preprocessing.normalize(df[["x", "y", "z"]])
    return df
