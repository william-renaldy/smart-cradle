import pickle
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import csv
import tensorflow as tf
import warnings
import sounddevice as sd
import wavio as wv
from pygame import mixer
import time
warnings.filterwarnings('ignore')



class TestPreprocess:
    def preprocess(self,songname):

        with open("scaler.pkl","rb") as m:
            scaler = pickle.load(m)


        cmap = plt.get_cmap('inferno')
        pathlib.Path(f'test_data').mkdir(parents=True, exist_ok=True)
        print(songname)
        filename = songname.split("/")[-1]
        y, sr = librosa.load(songname, mono=True, duration=5)
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB')
        plt.axis('off')
        plt.savefig(f'test_data/{filename[:-3].replace(".", "")}.png')

        header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
        for i in range(1, 21):
            header += f' mfcc{i}'
        header = header.split()


        file = open('testdataset.csv', 'w', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(header)

        y, sr = librosa.load(songname, mono=True, duration=30)
        rmse = librosa.feature.rms(y=y)[0]
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        file = open('testdataset.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())



        data = pd.read_csv('testdataset.csv')

        data = data.drop(['filename'],axis=1)#Encoding the Labels
        yy = scaler.transform(np.array(data, dtype = float))        

        return yy


    
class Model:
    def __init__(self) -> None:
        try:
            self.model = tf.keras.models.load_model('my_model')
        except:
            pass


    def predict(self,x):
        y = self.model.predict(x)

        print(y)
        if (np.argmax(y) == 1):
            return "Not crying"
        else:
            return "Crying"







class Audio:
    def __init__(self):
        self.freq = 44100
        self.duration = 10

    def recorder(self):
        recording = sd.rec(int(self.duration * self.freq),
				samplerate=self.freq, channels=2)

        sd.wait()

        wv.write("recording1.wav", recording, self.freq, sampwidth=2)


class PlayAudio:
    def __init__(self) -> None:
        mixer.init()
        mixer.music.load('audio.mp3')
        mixer.music.play()
        time.sleep(10)
        mixer.music.stop()
