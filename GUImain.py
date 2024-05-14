#GUI main.py using tkinter
from tkinter import *
import threading
from network_models import katalina, splines
import random
import math
import matplotlib.pyplot as plt
import time
import os

class GUI:
    def __init__(self, master):
        self.master = master
        self.master.geometry("300x300")
        self.master.title("Control")
        self.button1 = Button(self.master, text="start", command=self.start).pack()
        self.button2 = Button(self.master, text="pause", command=self.pause).pack()
        self.button3 = Button(self.master, text="resume", command=self.resume).pack()
        self.button4 = Button(self.master, text="test_drive", command=self.test_drive).pack()
        self.button5 = Button(self.master, text="plot training progress", command=self.plot).pack()
        self.button6 = Button(self.master, text="erase history", command=self.erase).pack()

        structure = [4, 2, 1, 1]
        batch_size = 400
        X = []
        Y = []
        for i in range(0, batch_size):
            x1 = random.random()
            x2 = random.random()
            x3 = random.random()
            x4 = random.random()
            X.append([x1, x2, x3, x4])
            this_output = math.exp(math.sin(x1**2 + x2 ** 2) + math.sin(x3**2 + x4**2))
            Y.append([this_output])

        self.AI = katalina.NN(structure, train_inputs=X, train_outputs=Y)
        #global the data
        self.X = X
        self.Y = Y


    def start(self):
        #prepare data
        def run():
            results = self.AI.train(tolerance=0.02)
            return(results)

        p = threading.Thread(target=run).start()

    def pause(self):
        self.AI.pause = True

    def resume(self):
        current_hyperparameters = self.AI.spc
        self.AI.pause = False
        def run():
            self.spc = self.AI.train(preload_hyperparameters=current_hyperparameters)

        p = threading.Thread(target=run).start()

    def test_drive(self):
        current_hyperparameters = self.AI.spc
        test_out = self.AI.forward_propagate(current_hyperparameters, self.X)
        test_Y = []
        for i in test_out:
            test_Y.append(i[0])

        test_X = []
        for i in self.X:
            test_X.append(i[0])

        real_Y = []
        for i in self.Y:
            real_Y.append(i[0])

        fig = plt.Figure()
        tests = plt.scatter(test_X, test_Y)
        real = plt.scatter(test_X, real_Y)

        plt.show()

    def plot(self):
        def run():
            os.system("python3 plot_training.py")
        threading.Thread(target=run).start()

    def erase(self):

        os.system("rm -r history.csv")

os.system("rm -r history.csv")
root = Tk()
GUI(root)
root.mainloop()
