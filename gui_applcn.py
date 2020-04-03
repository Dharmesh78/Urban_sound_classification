from tkinter import Tk, Label, Button
from tkinter import filedialog
import librosa
import tkinter as tk
import numpy  as np
from keras import models
from playsound import  playsound

class_model=models.load_model("sound_classifier_model")


def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=49)
        mfccsscaled = np.mean(mfccs.T, axis=0)

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return mfccsscaled

labels=["Air Conditioner","Car Horn","Children Playing","Dog Bark","Drilling","Engine Idling","Gunshot","Jackhammer","Siren","Street Music"]
def get_class(value):
    return labels[value]

class MyFirstGUI:
    def __init__(self, master):
        self.master = master
        master.title("URBAN SOUND CLASSIFICATION---DHARMESH SINGH")
        self.msg="Load the sound and predict the class"
        self.label = Label(master, text=self.msg)
        self.label.pack(pady=10)

        self.load_sound_button = Button(master, text="Load", command=self.load_sound, bg="light green", fg="black")
        self.load_sound_button.pack(pady=10)

        self.fileLabel = Label(master, text="no file selected", bg="white", fg="black")
        self.fileLabel.pack(pady=10)

        self.play_sound_button = Button(master, text="PLAY TO LISTEN", command=self.play_sound, bg="light green",
                                        fg="black")
        self.play_sound_button.pack(pady=10)

        self.playLabel = Label(master, text="", bg="white", fg="black")
        self.playLabel.pack(pady=10)

        self.predict_button = Button(master, text="Predict", command=self.predict_class,bg="yellow", fg="black")
        self.predict_button.pack(pady=10)

        self.outlabel = Label(master,bg="white", fg="black")
        self.outlabel.pack(pady=10)

        self.close_button = Button(master, text="Close", command=master.quit, bg="red", fg="white")
        self.close_button.pack(pady=10)

    def load_sound(self):
        root.filename = filedialog.askopenfilename(initialdir="/PycharmProjects/NITtrichy/SOUND-CLASSFICATION/UrbanSound8K/audio", title="Select file",
                                                       filetypes=(("wav files", "*.wav"), ("all files", "*.*")))
        l=root.filename.split("/")
        self.fileLabel["text"]=l[-1]+" is loaded"

    def play_sound(self):
        l = root.filename.split("/")
        self.playLabel.config(text=l[-1]+" is played!!")
        playsound(root.filename)




    def predict_class(self):
        features = extract_features(root.filename)
        l = root.filename.split("/")
        features = np.reshape(features, (1, 7, 7, 1))
        res = class_model.predict_classes(features)
        sound_class = get_class(res[0])
        self.outlabel.config(text="CLASS: " + sound_class)
        print("%s %d %s" % (l[-1], res[0], sound_class))


root = Tk()
root.minsize(600,200)
root.configure(background='light blue')
my_gui = MyFirstGUI(root)
root.mainloop()