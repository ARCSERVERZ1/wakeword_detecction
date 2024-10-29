import os
import librosa.display
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

class WakeWordBuilder:
    def __init__(self, main_path, wake_word, model_path, noise_path, buffer , model):

        self.main_path = main_path + '/'
        self.model_path = self.main_path + model_path
        self.wake_word = self.main_path + wake_word
        self.noise_path = self.main_path + noise_path
        self.buffer = self.main_path + buffer
        self.no_of_samples = 15
        self.fs = 44100
        self.seconds = 2
        self.model = self.model_path+'/'+model+'.h5'

        if not os.path.exists(self.noise_path): os.makedirs(self.noise_path)
        if not os.path.exists(self.model_path): os.makedirs(self.model_path)
        if not os.path.exists(self.buffer): os.makedirs(self.buffer)
        if not os.path.exists(self.wake_word): os.makedirs(self.wake_word)

        self.path_list = {
            'model_path': self.model_path,
            'wake_word': self.wake_word,
            'noise_path': self.noise_path,
            'buffer': self.buffer,
        }

    def get_path_list(self):
        return self.path_list

    def record_data(self, type, params):

        try:

            print(f"Recording started for {params[0]}")
            recording = sd.rec(int(self.fs * self.seconds), samplerate=self.fs, channels=2)
            sd.wait()
            write( self.path_list[type] + "/" + type + '_' + str(params) + ".wav", self.fs, recording)
            print("saved")
        except:
            print("Error")

    def preprocessing(self):
        all_data = []
        list_of_files = {
            0: [self.noise_path + '/' + file for file in os.listdir(self.noise_path + '/')],
            1: [self.wake_word + '/' + file for file in os.listdir(self.wake_word + '/')]
        }

        if len(list_of_files[0]) >= self.no_of_samples and len(list_of_files[1]) >= self.no_of_samples:

            for class_label, file_paths in list_of_files.items():
                for file in file_paths:
                    data, sample_rate = librosa.load(file)
                    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
                    mfcc_processed = np.mean(mfccs.T, axis=0)
                    all_data.append([mfcc_processed, class_label])
                print(f"data preprocessing for class {class_label} completed")

            return [True, pd.DataFrame(all_data, columns=["feature", "class_label"])]
        else:
            return [False]

    def train_model(self, model_name):

        output = self.preprocessing()

        if output[0]:
            X = output[1]["feature"].values
            X = np.concatenate(X, axis=0).reshape(len(X), 40)

            y = np.array(output[1]["class_label"].tolist())
            y = to_categorical(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = Sequential([
                Dense(256, input_shape=X_train[0].shape),
                Activation('relu'),
                Dropout(0.5),
                Dense(256),
                Activation('relu'),
                Dropout(0.5),
                Dense(2, activation='softmax')
            ])

            print(model.summary())

            model.compile(
                loss="categorical_crossentropy",
                optimizer='adam',
                metrics=['accuracy']
            )

            print("Model Score: \n")
            model.fit(X_train, y_train, epochs=1000)
            print(f"{self.model_path}/{model_name}.h5")
            model.save(f"{self.model_path}/{model_name}.h5")
            score = model.evaluate(X_test, y_test)
            print(score)

        else:
            print("not enough data to train model")

    def predict(self):
        self.model = load_model(self.model)
        print(f'model loaded {self.model}')
        while True:
            myrecording = sd.rec(int(self.seconds * self.fs), samplerate=self.fs, channels=2)
            sd.wait()
            prediction_recording = self.path_list['buffer'] + "/" +  "predict.wav"
            write(prediction_recording , self.fs, myrecording)
            audio, sample_rate = librosa.load(prediction_recording)
            mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfcc_processed = np.mean(mfcc.T, axis=0)
            prediction = self.model.predict(np.expand_dims(mfcc_processed, axis=0))
            print(prediction[0][0] * 100, prediction[0][1] * 100)



if __name__ == '__main__':
    WakeWordBuilder().record_data('wake_word')
