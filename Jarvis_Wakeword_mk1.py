import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import time
import pyttsx3


class initialiseJarvis:
    def __init__(self, model_path):
        self.model_path = model_path
        self.listen = True
        self.voice_engine = None
        self.fs = 44100
        self.seconds = 2
        self.filename = 'prediction.wav'
        self.model = load_model(self.model_path)
        self.prediction = False
        self.set_voice()
        self.start_listening()

    def set_voice(self):
        self.voice_engine = pyttsx3.init()
        self.voice_engine.setProperty('voice', 'english+f3')  # You may need to adjust this to find a suitable voice
        self.voice_engine.setProperty('rate', 230)

    def response(self, message):
        self.voice_engine.say(message)
        self.voice_engine.runAndWait()

    def start_listening(self):
        while True:
            myrecording = sd.rec(int(self.seconds * self.fs), samplerate=self.fs, channels=2)
            sd.wait()
            write(self.filename, self.fs, myrecording)
            audio, sample_rate = librosa.load(self.filename)
            mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfcc_processed = np.mean(mfcc.T, axis=0)
            prediction = self.model.predict(np.expand_dims(mfcc_processed, axis=0))
            print(prediction[0][0] * 100, prediction[0][1] * 100)
            if prediction[0][1]*100 > 80.999 and prediction[0][0]*100 < 0.1:
                print(prediction[0][0] * 100, prediction[0][1] * 100)
                self.prediction = True
                break
        if self.prediction: self.respond_for_call()

    def respond_for_call(self):
        self.prediction = False
        self.response(" at your service sir")
        time.sleep(5)
        self.start_listening()


if __name__ == '__main__':
    obj = initialiseJarvis('saved_model/WWD.h5')
