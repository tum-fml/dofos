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

# with open('D_7conditions.pkl', 'rb') as m:
#     dt = joblib.load(m)

def main():
    print()
    print("--------------------------------------------")
    print()
    print("Mustererkennung EFG220")
    print()
    print("--------------------------------------------")

    # Validierung
    alles = pd.read_csv("C:/Users/Chamou/Desktop/Daten Validierung/datalog4.txt", sep=' ', names=['x', 'y', 'z']
                     , usecols=[0, 1, 2])

    alles = c.med_filter(alles, 3)
    alles = c.clean_data(alles)
    # Trainingsdaten
    # df = pd.read_csv("C:/Users/Chamou/PycharmProjects/01_Mustererkennung_Beschleunigungssensor/Datenbank_Betriebszustände/"
    #                  "03_aufschlagen_zinke_untergrund/sample_pattern13.txt", sep=' ', names=['x', 'y', 'z'], usecols=[0, 1, 2])
    # print(len(alles))

    with open('R_3conditions.pkl', 'rb') as m:
        dt = joblib.load(m)

    # df = alles.iloc[58500:58700]
    # print(df)

    for i in range(0, (len(alles)-1000), 20):
    # for i in range(2):
        df = alles.iloc[0+i:100+i]

        # plt.plot(df)
        # plt.show()

        features = [0, 2, 5, 6]
        feature_matrix = np.zeros(3 * (len(features)))
        extraction = c.features(len(features), feature_matrix, 1)

        # df = c.med_filter(df, 9)
        # df = c.clean_data(df)

        l = 0
        n = 1
        if 0 in features:
            extraction.peak_to_peak(df, n, l)
            l += 3
        if 1 in features:
            extraction.euclidean_Dist(df, n, l)
            l += 3
        if 2 in features:
            extraction.fft(df, n, l)
            l += 3
        if 3 in features:
            extraction.min(df, n, l)
            l += 3
        if 4 in features:
            extraction.max(df, n, l)
            l += 3
        if 5 in features:
            extraction.var(df, n, l)
            l += 3
        if 6 in features:
            extraction.std(df, n, l)
            l += 3

        condition = dt.predict(feature_matrix.reshape(1,-1))
        print("Condition is: ", condition, " @ ", "i = ", i)


if __name__ == "__main__":
    main()