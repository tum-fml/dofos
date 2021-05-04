################################################################################
#import RPi.GPIO as GPIO

#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import time
from time import sleep
import serial  # USB-Kommunikation
import numpy as np
import pandas as pd
import datetime
import can  # CAN-Kommunikation
import joblib  # Trainierte ML-Modelle speichern/aufrufen.

# scikit-learn.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

from sklearn.svm import SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

#time.sleep(45)
#GPIO.setwarnings(False)
#GPIO.setmode(GPIO.BCM)

#sleep(10)

# ID's der einzelnen Nachrichten
id3 = 0x19C  # Erkannter Betriebszustand - Node-ID: 0x1C
id33 = 0x71C  # Heartbeat Erkannter Betriebszustand

id4 = 0x19D  # Kritischer Betriebszustand - Node-ID: 0x1D
id44 = 0x71D  # Heartbeat Kritischer Betriebszustand


os.system('sudo ifconfig can0 down')  # CAN Verbindung schließen falls eine geöffnet.

# Starten der CAN Verbindung
os.system('sudo ip link set can0 type can bitrate 250000')
os.system('sudo ifconfig can0 up')

can0 = can.interface.Bus('can0', bustype='socketcan_ctypes')


# Erstellen der CAN Nachricht
def msg_send(x, zusammenfassen):

    # Erkannter Betriebszustand - Node-ID: 0x1C
    # VORLÄUFIGE Legende:
    # 1 - Heben/Senken | 2 - Aufschlagen/Schleifen | 3 - Stehen | 4 - Fahren
    msg3 = can.Message(arbitration_id=id3, data=[x], extended_id=False)
    can0.send(msg3)

    # Heartbeat Erkannter Betriebszustand - Node-ID: 0x1C
    msg33 = can.Message(arbitration_id=id33, data=[5], extended_id=False)
    can0.send(msg33)


    # Kritischer Betriebszustand - Node-ID: 0x1D
    # VORLÄUFIGE Legende
    # 2 - Aufschlagen/Schleifen
    if x == 2 and zusammenfassen > 4:  # zusammenfassen unten bei der klassifikation...
        msg4 = can.Message(arbitration_id=id4, data=[5], extended_id=False)  # Auf 5 setzen. Tomay Steuerung so eingestellt. 
        can0.send(msg4)
        print(msg4)
    else:
        msg4 = can.Message(arbitration_id=id4, data=[0], extended_id=False)  # 0 einfach nichts kritisches passiert. 
        can0.send(msg4)

    # Heartbeat Kritischer Betriebszustand - Node-ID: 0x1D
    msg44 = can.Message(arbitration_id=id44, data=[5], extended_id=False)
    can0.send(msg44)

# Spielt kurzen Ton ab. Allerdings bleibt das Skript stehen bis Ton gespielt. Parallel? 
def play():
    duration = 0.1  # seconds
    freq = 440  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

# Euklidische Distanz zum Referenzsignal. Für x, y = 0.
# Nicht wirklich Aussagekräftig. Keine wirkliche Verbesserung der Modelle. ÜBERARBEITEN!
def euclidean_Dist(df):
    pattern_length = len(df)
    xy_reference = np.zeros(pattern_length)

    # Scaled
    x = np.linalg.norm(df['x_scaled']-xy_reference)
    y = np.linalg.norm(df['y_scaled']-xy_reference)
    z = np.linalg.norm(df['z_scaled']-xy_reference)

    # X[0, 0] = x + y + z

    X[0, 0] = x
    X[0, 1] = y
    X[0, 2] = z

# Peak to peak
# Für Zustände wie Aufschlagen sehr gut. Testen ohne Betrag. Unterscheidung bei bsp. Heben / Senken quasi nicht gegeben. Überarbeiten.
def peak_to_peak(df):

    # X[0, 1] = abs(max(df['x']) - min(df['x'])) + abs(max(df['y']) - min(df['y'])) + abs(max(df['z']) - min(df['z']))
    X[0, 3] = abs(max(df['x']) - min(df['x']))
    X[0, 4] = abs(max(df['y']) - min(df['y']))
    X[0, 5] = abs(max(df['z']) - min(df['z']))

# Fast Fourier Transformation
def fft(df):

    x = np.abs(sum(np.fft.fft(df['x_scaled'])))
    y = np.abs(sum(np.fft.fft(df['y_scaled'])))
    z = np.abs(sum(np.fft.fft(df['z_scaled'])))

    # X[0, 2] = x + y + z

    X[0, 6] = x
    X[0, 7] = y
    X[0, 8] = z

# Öffnen der gespeicherten Modelle zur anschließenden Klassifikation.
# Darauf Achten welche Modelle geladen werden. Ansonsten passt die zuteilung der Zustandsnummern nicht!!!
# Evtl. Verknüpfung mit Trainingsskript erstellen. Variabler gestalten. 
#with open('/home/pi/Desktop/dt_vortrag.pkl', 'rb') as fid:
with open('/home/pi/FML/Beschleunigungssensor/dt_vortrag.pkl', 'rb') as fid:
    clf = joblib.load(fid)

# Feature Vektor. Momentan 9 Values. Definiert in einzelnen Funktionen. EukD - PtP - FFT
X = np.zeros((1, 9))


# 
size_array = 100
index_array = size_array - 1

# k siehe fifo_buffer()
k = 0

# Empty Arrays for FIFO Buffer
x = np.zeros(size_array)
y = np.zeros(size_array)
z = np.zeros(size_array)

# Python Serial Communication Port
if os.path.exists('/dev/ttyUSB0'):
    port = '/dev/ttyUSB0'
else:
    port = '/dev/ttyUSB1'
    
#########################################
#########################################
#
# USB 0 Unten - 1 Oben wieder ändern wenn BS und Baro nicht gleichzeitig am laufen sind!
#
##########################################
##########################################
    
#port = '/dev/ttyUSB1'

ser = serial.Serial(port=port, baudrate=115200)  # without timeout!

# Filling up FIFO Arrays. Predefined at the beginning of this Code.
def fifo_buffer():
    
    # C++ Code sichern. GitHub nachziehen!!!!
    data = ser.readline()  # Get out raw output of ESP32-WROOM. Check C++ Code flashed on device.
    separated = np.fromstring(data, sep=' ')  # It´s working. Why? :D (Got Bytes instead of string.)
    # Seperated nicht notwendig. Auslesen mit UTF-8. Kein numpy notwendig. Bei Zeit ÜBERARBEITEN.

    global k

    if k > 1:  # ersten beiden Beschleunigungstupel beim Start des Streams ignorieren. Können fehlerhaft sein.

        global x, y, z, xyz  # Need GLOBAL to get pre-allocated Array. Werte an Funktion übergeben. Jesus. ÜBERARBEITEN.

        x_update = np.roll(x, -1)  # roll() function rotates Values like in a Ring-Buffer
        y_update = np.roll(y, -1)  # Update des zu betrachteten Signals 
        z_update = np.roll(z, -1)
        
        # Verhindert das bei fehlerhaften Einträgen das Skript nicht abbricht.
        # Sollte mit k oben vermieden werden. Testen und wenn möglich löschen. 
        if len(separated) == 3:
            x_update[index_array] = separated[0]
            y_update[index_array] = separated[1]
            z_update[index_array] = separated[2]
        if len(separated) == 2:
            x_update[index_array] = separated[0]
            y_update[index_array] = separated[1]
            z_update[index_array] = 1
        if len(separated) == 1:
            x_update[index_array] = separated[0]
            y_update[index_array] = 0
            z_update[index_array] = 1
        if len(separated) == 0:
            x_update[index_array] = 0
            y_update[index_array] = 0
            z_update[index_array] = 1
        
        # Fenster Update. Speichervorgang nachlesen. Speichereffizients!
        x = x_update
        y = y_update
        z = z_update

    k += 1

# Laufzeit Klassifikation 
#start = time.perf_counter()

def main():
    # i bremst Klassifikation aus. Ohne wird bei jedem neuen Tupel klassifiziert. ~400Hz Aufnahme!
    i = 0
    zusammenfassen = 0
    
    msg_send(2, 5) # Soll bei Start einmal vibrieren
    sleep(1)
    msg_send(2, 5)
    
    while True:

        fifo_buffer()
        i += 1
        if i == 60:
            xx = np.reshape(x, (-1, 1))
            yy = np.reshape(y, (-1, 1))
            zz = np.reshape(z, (-1, 1))

            df = pd.DataFrame(data=xx)
            df.columns = ['x']
            df['y'] = pd.DataFrame(data=yy)
            df['z'] = pd.DataFrame(data=zz)

            # Normalization
            min_max_scaler = preprocessing.MinMaxScaler()

            x_scaled = min_max_scaler.fit_transform(xx)
            y_scaled = min_max_scaler.fit_transform(yy)
            z_scaled = min_max_scaler.fit_transform(zz)

            df['x_scaled'] = pd.DataFrame(x_scaled[:, 0])
            df['y_scaled'] = pd.DataFrame(y_scaled[:, 0])
            df['z_scaled'] = pd.DataFrame(z_scaled[:, 0])

            clean_dataset(df)
            euclidean_Dist(df)
            peak_to_peak(df)
            fft(df)

            # Classification
            zustand = clf.predict(X)
            print(zustand + ' ' + '---' + ' ' + str(datetime.datetime.now()))
            #if zustand == 'aufschlagen/schleifen':
            #    print(zustand + ' ' + '---' + ' ' + str(datetime.datetime.now()))
            #    #background_thread = Thread(target=play)
            #    #background_thread.start()
            #    #background_thread.join()
            #    #process = multiprocessing.Process(target=play)
            #    #process.start()
            #else:
            #    print(zustand + ' ' + '---' + ' ' + str(datetime.datetime.now()))
            
            # Kontrollstruktur zwecks Signal an Steuerung.
            # Aufschleigen / Schleifen 5 mal in Folge dann Signal an Steuerung.
            # Struktur überdenken und Feintuning!
            if zustand == "heben/senken":
                q = 1
                zusammenfassen -= 1
            elif zustand == "aufschlagen/schleifen":
                q = 2
                zusammenfassen += 1
                #GPIO.setmode(GPIO.BCM)
                #GPIO.setup(23, GPIO.OUT)
                #sleep(0.1)
                #GPIO.cleanup()
            elif zustand == "stehen":
                q = 3
                zusammenfassen -= 1
            elif zustand == "fahren":
                q = 4
                zusammenfassen -= 1
            else:
                q = 0
            
            if zusammenfassen < 0 or zusammenfassen > 8:
                zusammenfassen = 0
                
            msg_send(q, zusammenfassen)
            #print(zusammenfassen)
            


        if i > 60:
            i = 0


if __name__ == '__main__':
    main()

    #finish = time.perf_counter()
    #print(f'Finished in {round(finish-start, 2)} seconds')
