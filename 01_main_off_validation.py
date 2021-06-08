# ######################################################################################################################
#
# Projekt: Mustererkennung anhand Beschleunigungsdaten der Gabelzinke
#
# Projektleitung: Leohnard Feiner TUM
#
# Autor Code: Filippos Chamoulias TUM
#
# fml - Lehrstuhl für Fördertechnik Materialfluss Logistik
#
# ######################################################################################################################
import os

import classes as c

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import joblib

from time import perf_counter

def main():
    print()
    print("--------------------------------------------")
    print()
    print("Mustererkennung EFG220")
    print()
    print("--------------------------------------------")

    # Validierung
    # Insert the path of the validation set
    alles = pd.read_csv("C:/Users/Chamou/LRZ Sync+Share/01_Paper/04_Daten_Validierung/datalog4.txt", sep=' ', names=['x', 'y', 'z']
                     , usecols=[0, 1, 2])

    print(len(alles))

    df = c.med_filter(alles, 3)
    df = c.clean_data(alles)

    df = alles

    # Preparing Plot for later vizualisation 
    fig, ax = plt.subplots()
    ax.plot(df, color='k')

    # The same features must be used as for training
    features = [0, 4, 5]

    # Load ML Model
    with open('R_3conditions.pkl', 'rb') as m:
        classifier = joblib.load(m)


    # Select window size
    non_overlapping = 50

    x = non_overlapping
    steps = non_overlapping


    start = 0
    stop = len(df)


    for i in range(start, stop, steps):

        window = df.iloc[0+i:x+i]

        feature_matrix = np.zeros(3 * (len(features)))
        extraction = c.features(len(features), feature_matrix, 1)

        # df = c.med_filter(df, 3)
        # df = c.clean_data(df)

        l = 0
        n = 1
        if 0 in features:
            extraction.peak_to_peak(window, n, l)
            l += 3
        if 1 in features:
            extraction.euclidean_Dist(df, n, l)
            l += 3
        if 2 in features:
            extraction.fft(window, n, l)
            l += 3
        if 3 in features:
            extraction.min(window, n, l)
            l += 3
        if 4 in features:
            extraction.max(window, n, l)
            l += 3
        if 5 in features:
            extraction.var(window, n, l)
            l += 3
        if 6 in features:
            extraction.std(window, n, l)
            l += 3
        if 7 in features:
            extraction.skewness(window, n, l)
            l += 3
        if 8 in features:
            extraction.kurtosis(window, n, l)
            l += 3

        condition = classifier.predict(feature_matrix.reshape(1,-1))
        # print("Condition is: ", condition, " @ ", "i = ", i)


        if condition == 0:
            color = 'blue'
            a = 0.3
        if condition == 1:
            color = 'red'
            a=0.7
        if condition == 2:
            color = 'fuchsia'
            a = 0.3
        if condition == 3:
            color = 'blue'
            a = 0.3
        if condition == 4:
            color = 'orange'
            a = 0.3
        if condition == 5:
            color = 'chartreuse'
            a = 0.3
        if condition == 6:
            color = 'cyan'
            a = 0.3
        if condition == 7:
            color = 'silver'
            a = 0.3
        ax.axvspan(0 + i,x + i, alpha=a, color=color)
    fontzize = 14
    plt.xlabel('Datapoints', fontsize=fontzize)
    plt.ylabel('g (m/s²)', fontsize=fontzize)
    plt.locator_params(nbins=4)

    plt.yticks(fontsize=fontzize)
    plt.xticks(fontsize=fontzize)
    plt.show()

if __name__ == "__main__":
    main()
