import customtkinter as ctk
import WakeWordBuilder as wwd
import threading
import json, os, glob
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import librosa.display


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.ww_entry = None
        self.wwd = None
        self.tabview = None
        self.title("Sunday !")
        self.resizable(False, False)  # Prevent resizing the window
        self.config_data = None
        self.screen_width = 350
        self.screen_height = 500
        self.geometry(f"{self.screen_width}x{self.screen_height}")
        self.record_thread = None
        self.current_rec = []
        self.startup_operations()

    def startup_operations(self):
        self.config_data = json.loads(open('sunday_config.json').read())
        if self.config_data['main_path'].lower() == 'current': self.config_data['main_path'] = os.getcwd()
        self.wwd = wwd.WakeWordBuilder(self.config_data['main_path'], self.config_data['wake_word_folder'],
                                       self.config_data['model_folder'], self.config_data['noise_folder'],
                                       self.config_data['buffer_folder'] , self.config_data['wake_word_model'] )
        self.path_data = self.wwd.get_path_list()
        print(self.path_data)
        self.create_ui()

    def create_ui(self):
        # Create the tabview
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(expand=True, fill="both")
        self.tabview.add("record_data")
        self.tabview.add("Training")

        wake_word_files = glob.glob(
            os.path.join(self.config_data['main_path'] + '/' + self.config_data['wake_word_folder'], '*.wav'))
        noise_files = glob.glob(
            os.path.join(self.config_data['main_path'] + '/' + self.config_data['noise_folder'], '*.wav'))

        ww_record_frame = ctk.CTkFrame(self.tabview.tab("record_data"), fg_color='red')
        ww_record_frame.pack(side=ctk.TOP, fill=ctk.X, pady=10)

        ww_label = ctk.CTkLabel(ww_record_frame, text=f'Next Rec No:', text_color="black", font=("Arial", 12))
        ww_label.pack(side=ctk.LEFT, padx=10, pady=10, expand=True)

        self.ww_entry = ctk.CTkEntry(ww_record_frame, text_color="black", font=("Arial", 12), width=35)
        self.ww_entry.pack(side=ctk.LEFT, padx=10, pady=10, expand=True)
        self.ww_entry.insert(0, len(wake_word_files))

        ww_button2 = ctk.CTkButton(ww_record_frame, text="Rec. Wake Word",
                                   command=lambda: self.record_data('wake_word', self.ww_entry.get()))
        ww_button2.pack(side=ctk.BOTTOM, padx=10, pady=10)

        noise_record_frame = ctk.CTkFrame(self.tabview.tab("record_data"), fg_color='red')
        noise_record_frame.pack(side=ctk.TOP, fill=ctk.X, pady=10)

        noise_label = ctk.CTkLabel(noise_record_frame, text=f'Next Rec No:', text_color="black",
                                   font=("Arial", 12))
        noise_label.pack(side=ctk.LEFT, padx=10, pady=10, expand=True)

        self.noise_entry = ctk.CTkEntry(noise_record_frame, text_color="black", font=("Arial", 12), width=35)
        self.noise_entry.pack(side=ctk.LEFT, padx=10, pady=10, expand=True)
        self.noise_entry.insert(0, len(noise_files))

        noise_button2 = ctk.CTkButton(noise_record_frame, text="Rec. Noise",
                                      command=lambda: self.record_data('noise_path', self.noise_entry.get()))
        noise_button2.pack(side=ctk.BOTTOM, padx=10, pady=10)

        buffer_record_frame = ctk.CTkFrame(self.tabview.tab("record_data"), fg_color='red')
        buffer_record_frame.pack(side=ctk.TOP, fill=ctk.X, pady=10)

        buffer_button = ctk.CTkButton(buffer_record_frame, text="Rec. Test",
                                      command=lambda: self.record_data('buffer', 'show'))
        buffer_button.pack(side=ctk.BOTTOM, padx=10, pady=10)

        graph = ctk.CTkFrame(self.tabview.tab("record_data"))
        graph.pack(side=ctk.TOP, fill=ctk.X)

        fig = Figure(figsize=(3, 2), dpi=100)
        self.plot = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, master=graph)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(pady=20)

        train_frame = ctk.CTkFrame(self.tabview.tab("Training"), fg_color='red')
        train_frame.pack(side=ctk.TOP, fill=ctk.X, pady=10)

        start_train = ctk.CTkButton(train_frame, text="Start Training", command=self.start_train)
        start_train.pack(side=ctk.BOTTOM, padx=10, pady=10)

        start_train = ctk.CTkButton(train_frame, text="start predict", command=self.start_predict)
        start_train.pack(side=ctk.BOTTOM, padx=10, pady=10)

    def start_predict(self):
        threading.Thread(target=self.wwd.predict).start()

    def start_train(self):
        print("training started cmd received")
        threading.Thread(target=lambda: self.wwd.train_model('first_model')).start()

    def show_audio_data1(self):
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        self.plot.clear()
        self.plot.plot(x, y)  # Plot new data
        self.plot.set_title("Cosine Function")
        self.plot.set_xlabel("X axis")
        self.plot.set_ylabel("Y axis")
        self.canvas.draw()

    def show_audio_data(self):
        self.record_thread.join()
        try:
            data, sample_rate = librosa.load(self.path)
            time = librosa.times_like(data, sr=sample_rate)
            self.plot.clear()
            self.plot.plot(time, data)
            self.plot.set_title('Audio Waveform')
            self.plot.set_xlabel('Time (seconds)')
            self.plot.set_ylabel('Amplitude')
            self.canvas.draw()

            if self.current_rec[0] == 'wake_word':
                self.current_rec[1] = int(self.current_rec[1]) + 1
                self.ww_entry.delete(0, ctk.END)
                self.ww_entry.insert(0, self.current_rec[1])
            elif self.current_rec[0] == 'noise_path':
                self.current_rec[1] = int(self.current_rec[1]) + 1
                self.noise_entry.delete(0, ctk.END)
                self.noise_entry.insert(0, self.current_rec[1])
            else:
                pass

        except:
            print("Error in Show Audio data")

    def record_data(self, type, count):
        self.current_rec = [type, count]
        self.path = self.path_data[type] + "/" + type + '_' + count + ".wav"
        self.record_thread = threading.Thread(target=lambda: self.wwd.record_data(type, count))
        self.record_thread.start()
        print(self.record_thread)
        self.show_audio_data()


if __name__ == "__main__":
    app = App()
    app.mainloop()
