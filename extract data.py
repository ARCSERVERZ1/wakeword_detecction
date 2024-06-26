import os
import librosa
import  librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


sample = "wake_word/1.wav"
data,sample_rate = librosa.load(sample)

# plt.title("wave form")
# librosa.display.waveshow(data , sr=sample_rate)
# plt.show()

mfccs = librosa.feature.mfcc(y = data , sr = sample_rate , n_mfcc = 40)
print(mfccs.shape)

# plt.title('MFCC')
# librosa.display.specshow(mfccs,sr=sample_rate , x_axis='time')
# plt.show()
all_data = []
list_of_files = {
    0: [ 'background/'+file for file in os.listdir('background/') ] ,
    1: [ 'wake_word/'+file for file in os.listdir('wake_word/') ]
}


for class_label , file_paths in list_of_files.items():
    for file in file_paths:
        data , sample_rate = librosa.load(file)
        mfccs = librosa.feature.mfcc(y = data  ,sr = sample_rate , n_mfcc = 40)
        mfcc_processed = np.mean(mfccs.T , axis=0)
        all_data.append([mfcc_processed , class_label])
    print(f"all preprocedd for class {class_label}")


df = pd.DataFrame(all_data , columns=["feature" , "class_label"])

df.to_pickle("final_data/wake_word.csv")