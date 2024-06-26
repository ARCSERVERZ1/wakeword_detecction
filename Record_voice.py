import sounddevice as sd
from scipy.io.wavfile import write




def record_wake_word(save_path, n_times=100):
    input("Press enter to start recording")
    for i in range(n_times):
        fs = 44100
        seconds = 2
        myrecording = sd.rec(int(fs*seconds),samplerate=fs,channels=2)
        sd.wait()
        write(save_path+str(i)+".wav",fs,myrecording)
        input(f"press enter to start next recording {i}")

def record_background(save_path , n_times = 100):
    input("Press enter to start recording")
    for i in range(n_times):
        fs = 44100
        seconds = 2
        print("start")
        myrecording = sd.rec(int(fs * seconds), samplerate=fs, channels=2)

        sd.wait()
        write(save_path + str(i) + ".wav", fs, myrecording)
        print(f"starting next recording {i}")



# record_wake_word('wake_word/')

record_background('background/')



