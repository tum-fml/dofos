# ######################################################################################################################
#
# Skript zum antrainieren von Mustererkennungsmodellen.
#
# Datenbank mit Mustern zu den einzelnen Betriebsvorgängen muss vorhanden sein. (Supervised Ansatz)
#
# ######################################################################################################################
import os
from collections import Counter

import classes as c

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import joblib

# ######################################################################################################################
condition = {1: 'heben',
             2: 'senken',
             3: 'aufschlagen_zinke',
             4: 'aufschlagen_hubgeruest',
             5: 'schleifen',
             6: 'stehen',
             7: 'fahren',
             0: 'andere'}
# ######################################################################################################################
# # Set up Data
# # Get all Data together and attache label
# # Filter (median) and clean data from NaN and Inf
# # Scale (min-max) for later training of knn etc.

# Get folder names in database
folder = [name for name in os.listdir("Datenbank_Betriebszustände")]

dataframe_list = []  # All pattern in one list. Every list object is an pandas dataframe.
labels = []  # List of labels for every pattern. Predefined.

# Set up given data with labels
for i in range(len(folder)):

    label = folder[i][0:2]
    operating_condition = folder[i][3:]
    path = "Datenbank_Betriebszustände/" + folder[i]

    list_dataframes = c.data(path=path, label=label, operating_condition=operating_condition)
    list_dataframes.list(dataframe_list=dataframe_list, labels=labels)

print(dataframe_list[188])
# median_window_size = 9  # Window size for median filter.

# Median filter and cleaning data (Nan, Inf, etc.)
for i in range(len(dataframe_list)):
    # dataframe_list[i] = c.med_filter(dataframe_list[i], median_window_size)
    dataframe_list[i] = c.clean_data(dataframe_list[i])
    # print(dataframe_list[i][~dataframe_list[i].applymap(np.isreal).all(1)])
    # print(i)

# Nochmals CHECKEN !!!!!!!!!!!!!
for i in range(len(dataframe_list)):
    # dataframe_list[i] = c.min_max_scaler(dataframe_list[i])
    dataframe_list[i] = c.normalize(dataframe_list[i])

print(dataframe_list[188])

print()
print("Dataframes in List: " + str(len(dataframe_list)))
print("Labels in List: " + str(len(labels)))
myset = list(set(labels))
counter = Counter(labels)
print(myset)
print(counter)
print()
print()
print(type(labels))
labels = [x if x == 3 or x == 5 else 0.0 for x in labels ]
print(labels)
# print("---------------------------------------------------------")


# ######################################################################################################################
# # Features Extraktion for each pattern
# # Setting up in a dataframe for later modell training.

#
# 1: peak_to_peak, 2: euclidian_distance, 3: ........
#

# features = 3
# Select the desired Features from the top selection and put the numbers into the array.
features = [0, 2, 5, 6]
# features = [1, 2, 3, 4, 5]
feature_matrix = np.zeros((len(dataframe_list), 3*(len(features))))

extraction = c.features(len(features), feature_matrix, 2)
# print("--------------------------------------------------------")
# print(feature_matrix)

l = 0
if 0 in features:
    for n in range(len(dataframe_list)):
        extraction.peak_to_peak(dataframe_list[n], n, l)
    l += 3
if 1 in features:
    for n in range(len(dataframe_list)):
        extraction.euclidean_Dist(dataframe_list[n], n, l)
    l += 3
if 2 in features:
    for n in range(len(dataframe_list)):
        extraction.fft(dataframe_list[n], n, l)
    l += 3
if 3 in features:
    for n in range(len(dataframe_list)):
        extraction.min(dataframe_list[n], n, l)
    l += 3
if 4 in features:
    for n in range(len(dataframe_list)):
        extraction.max(dataframe_list[n], n, l)
    l += 3
if 5 in features:
    for n in range(len(dataframe_list)):
        extraction.var(dataframe_list[n], n, l)
    l += 3
if 6 in features:
    for n in range(len(dataframe_list)):
        extraction.std(dataframe_list[n], n, l)
    l += 3


# df = pd.DataFrame(data=feature_matrix)
# print(df.values)

# df['activity'] = labels
# df['activityID'] = labels
# df['activity'] = df['activityID'].apply(condition.get)
# print(df['activity'])
# print(df.values)
# ######################################################################################################################
# Model Training

# model = c.training(features=df.values, labels=labels)
model = c.training(features=feature_matrix, labels=labels)

knn = model.knn(5)

svm = model.svm()

dt = model.dt()

rf = model.rf()

# ######################################################################################################################
# # Model Saving

save_model = c.save_model(knn, svm, dt, rf)
save_model.save()