# the main application with GUI in QT
import sys

try:
    from PyQt5.QtWidgets import (
        QApplication, QDialog, QMainWindow, QMessageBox
    )
    from PyQt5.uic import loadUi
    from PyQt5 import QtCore, QtGui, QtWidgets
except:
    print("missing pyqt5 dependency. Please install it")

import multiprocessing as mp
import threading
import os
import copy
import collections

try:
    import matplotlib.pyplot as plt
except:
    print("missing matplotlib")


class Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_inputs, self.model_outputs = self.load_data()
        self.setupUi(self)

    def setupUi(self, Frame):
        Frame.setObjectName("Frame")
        Frame.resize(708, 444)
        self.gridLayoutWidget = QtWidgets.QWidget(Frame)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 10, 701, 431))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.Layout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.Layout.setContentsMargins(0, 0, 0, 0)
        self.Layout.setObjectName("Layout")

        # group labels
        self.label_2 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.Layout.addWidget(self.label_2, 1, 0, 1, 1)

        self.label_5 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_5.setObjectName("label_5")
        self.Layout.addWidget(self.label_5, 3, 0, 1, 1)

        self.label_4 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_4.setObjectName("label_4")
        self.Layout.addWidget(self.label_4, 4, 0, 1, 1)

        self.label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label.setObjectName("label")
        self.Layout.addWidget(self.label, 2, 0, 1, 1)

        self.label_3 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.Layout.addWidget(self.label_3, 0, 0, 1, 1)

        self.label_6 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_6.setObjectName("label_6")
        self.Layout.addWidget(self.label_6, 5, 0, 1, 1)

        self.label_7 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_7.setObjectName("label_7")
        self.Layout.addWidget(self.label_7, 6, 0, 1, 1)

        self.label_8 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_8.setObjectName("label_8")
        self.Layout.addWidget(self.label_8, 5, 3, 1, 1)

        # buttons
        self.Start_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.Start_button.setObjectName("Start_button")
        self.Start_button.clicked.connect(self.start_training)
        self.Layout.addWidget(self.Start_button, 7, 0, 1, 1)

        self.Pause_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.Pause_button.setObjectName("pushButton")
        self.Pause_button.clicked.connect(self.pause)
        self.Layout.addWidget(self.Pause_button, 8, 0, 1, 1)

        self.Resume_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.Resume_button.setObjectName("Resume_button")
        self.Resume_button.clicked.connect(self.resume)
        self.Layout.addWidget(self.Resume_button, 8, 2, 1, 1)

        self.Run_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.Run_button.setObjectName("Run_button")
        self.Run_button.clicked.connect(self.run)
        self.Layout.addWidget(self.Run_button, 7, 2, 1, 1)

        self.Testdrive_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.Testdrive_button.setObjectName("Testdrive_button")
        self.Testdrive_button.clicked.connect(self.test_drive)
        self.Layout.addWidget(self.Testdrive_button, 9, 0, 1, 1)
        self.Plot_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.Plot_button.setObjectName("Plot_button")
        self.Plot_button.clicked.connect(self.graph)
        self.Layout.addWidget(self.Plot_button, 9, 2, 1, 1)
        self.Kill_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.Kill_button.setObjectName("Suicide")
        self.Kill_button.clicked.connect(self.suicide)
        self.Layout.addWidget(self.Kill_button, 10, 2, 1, 1)

        # entries
        self.LearningRate_entry = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.LearningRate_entry.setObjectName("lineEdit")
        self.LearningRate_entry.setText("0.001")
        self.Layout.addWidget(self.LearningRate_entry, 2, 2, 1, 1)

        self.comboBox = QtWidgets.QComboBox(self.gridLayoutWidget)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("Katalina")
        self.comboBox.addItem("Catherine")
        self.comboBox.addItem("TurboKAN1")
        self.comboBox.addItem("KAN_optimised")
        self.comboBox.addItem("Lerochka")
        self.Layout.addWidget(self.comboBox, 0, 2, 1, 1)

        self.Shape_entry = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.Shape_entry.setObjectName("Shape_entry")
        input_dim = str(len(self.model_inputs[0]))
        output_dim = str(len(self.model_outputs[0]))
        self.Shape_entry.setText(input_dim + "," + output_dim)
        self.Layout.addWidget(self.Shape_entry, 1, 2, 1, 1)

        self.LossTolerance_entry = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.LossTolerance_entry.setObjectName("LossTolerance_entry")
        self.LossTolerance_entry.setText("0.01")
        self.Layout.addWidget(self.LossTolerance_entry, 3, 2, 1, 1)

        self.Batchsize_entry = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.Batchsize_entry.setObjectName("Batchsize_entry")
        self.Batchsize_entry.setText("20")
        self.Layout.addWidget(self.Batchsize_entry, 4, 2, 1, 1)

        self.GridSize_entry = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.GridSize_entry.setObjectName("GridSize_entry")
        self.GridSize_entry.setText("10")
        self.Layout.addWidget(self.GridSize_entry, 5, 2, 1, 1)

        self.Order_entry = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.Order_entry.setObjectName("GridSize_entry")
        self.Order_entry.setText("10")
        self.Layout.addWidget(self.Order_entry, 6, 2, 1, 1)

        self.Momentum_entry = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.Momentum_entry.setObjectName("Momentum_entry")
        self.Momentum_entry.setText("0.01")
        self.Layout.addWidget(self.Momentum_entry, 5, 4, 1, 1)


        spacerItem1 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        spacerItem = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.Layout.addItem(spacerItem1, 11, 3, 1, 1)
        self.Layout.addItem(spacerItem, 11, 0, 1, 1)

        self.retranslateUi(Frame)
        QtCore.QMetaObject.connectSlotsByName(Frame)

    def retranslateUi(self, Frame):
        _translate = QtCore.QCoreApplication.translate
        Frame.setWindowTitle(_translate("Frame", "Frame"))
        self.label_2.setText(_translate("Frame", "Network Shape"))
        self.Start_button.setText(_translate("Frame", "Start Training"))
        self.label_5.setText(_translate("Frame", "Final loss tolerance"))
        self.Pause_button.setText(_translate("Frame", "Pause Training"))
        self.label_4.setText(_translate("Frame", "Batch Size"))
        self.LossTolerance_entry.setToolTip(_translate(
            "Frame", "The acceptable loss. The network stops training when this loss has been reached."))
        self.label.setText(_translate("Frame", "Initial learning rate"))
        self.label_3.setText(_translate("Frame", "Select Model"))
        self.Resume_button.setText(_translate("Frame", "Resume Training"))
        self.Run_button.setText(_translate("Frame", "Run"))
        self.Shape_entry.setToolTip(_translate("Frame", "Example: 4, 2, 1"))
        self.label_2.setToolTip(_translate(
            "Frame", "The first number is dimension of input vector, last is dimension of output vector. Middle numbers correspond to size of hidden layers."))
        self.comboBox.setToolTip(_translate(
            "Frame", "Select different network designs. KAN_optimised and turboKAN are made according to the paper; Catherine is single grid cubic spline; Katalina does split minibatch training."))
        self.Testdrive_button.setText(_translate("Frame", "Testdrive"))
        self.Plot_button.setText(_translate("Frame", "Plot Training Progress"))
        self.Kill_button.setText(_translate("Frame", "Kill"))
        self.label_6.setText(_translate("Frame", "Splines Gridcount"))
        self.label_7.setText(_translate("Frame", "Splines Order"))
        self.label_8.setText(_translate("Frame", "Momentum factor"))

    def start_training(self):
        def go():
            # firstly let's extract the user inputs
            shape = str(self.Shape_entry.text())
            try:
                shape = shape.split(",")
                structure = []
                for i in shape:
                    structure.append(int(i))
                print("Network shape: ", structure)
            except ValueError:
                print("Invalid input! Check your shape is formatted correctly")
            except Exception as e:
                print(e)

            # learning rate and loss tolerance are optional
            lr = float(self.LearningRate_entry.text())

            loss_tolerance = float(self.LossTolerance_entry.text())

            self.model_type = str(self.comboBox.currentText())

            if self.model_type == "Katalina":
                print("Selected model: Katalina")
                from network_models import katalina
                minibatch = int(self.Batchsize_entry.text())
                self.AI = katalina.NN(structure, learning_rate=lr,
                                      train_inputs=self.model_inputs, train_outputs=self.model_outputs)
                print("Initialisation complete, now training...")
                self.trained_hyperparameters = self.AI.train(
                    sub_batch_size=minibatch)

            elif self.model_type == "TurboKAN1":
                print("selected model: TurboKAN1")
                from network_models import turboKAN1
                Order = int(self.Order_entry.text())
                Grids = int(self.GridSize_entry.text())
                self.AI = turboKAN1.NN(structure, learning_rate=lr, order=Order, grids=Grids,
                                       train_inputs=self.model_inputs, train_outputs=self.model_outputs)
                self.trained_hyperparameters = self.AI.train()

            elif self.model_type == "KAN_optimised":
                print("selected model: KAN_optimised")
                from network_models import KAN_optimised
                Order = int(self.Order_entry.text())
                Grids = int(self.GridSize_entry.text())
                self.AI = KAN_optimised.NN(structure, learning_rate=lr, order=Order, grids=Grids,
                                           train_inputs=self.model_inputs, train_outputs=self.model_outputs)
                self.trained_hyperparameters = self.AI.train()

            elif self.model_type == "Catherine":
                print("Selected model: Catherine")
                from network_models import catherine
                self.AI = catherine.NN(structure, learning_rate=lr,
                                       train_inputs=self.model_inputs, train_outputs=self.model_outputs)
                print("Initialisation complete, now training...")
                self.trained_hyperparameters = self.AI.train()

            elif self.model_type == "Lerochka":
                print("Selected model: Lerochka")
                from network_models import lerochka
                minibatch = int(self.Batchsize_entry.text())
                p = float(self.Momentum_entry.text())

                self.AI = lerochka.NN(structure, learning_rate=lr,
                                      train_inputs=self.model_inputs, train_outputs=self.model_outputs)
                print("Initialisation complete, now training...")
                self.trained_hyperparameters = self.AI.train(
                    sub_batch_size=minibatch, momentum=p)

            else:
                print("Not implemented yet!")

        # performs the training in a separate thread so as not to crash the main window.
        threading.Thread(target=go).start()
        threading.Thread(target=self.log_parameters).start()
        threading.Thread(target=self.log_loss).start()

    def resume(self):
        self.AI.pause = False

        def go():
            self.AI.train(preload_hyperparameters=self.trained_hyperparameters)
        threading.Thread(target=go).start()

    def pause(self):
        self.AI.pause = True
        self.trained_hyperparameters = self.AI.spc

    def run(self):
        # load production inputs

        print("Reading data from test_inputs.csv")
        X = []
        with open("test_inputs.csv", "r") as f:
            next_line = str(f.readline())
            while next_line != "":
                line = next_line.split(",")
                this_x = []
                for i in line:
                    this_x.append(float(i))
                X.append(this_x)
                next_line = f.readline()

        # let the AI work on our test inputs
        def go():
            print("Unseen data loaded, now computing")
            output = 0
            if self.model_type == "Katalina" or self.model_type == "Catherine" or self.model_type == "Lerochka":
                self.trained_hyperparameters = self.AI.spc
                output = self.AI.forward_propagate(
                    self.trained_hyperparameters, X)

            if self.model_type == "TurboKAN1" or self.model_type == "KAN_optimised":
                c = self.AI.spc
                w = self.AI.w
                output = self.AI.forward_propagate(c, w, X)

            print("Calculation complete, saving output")
            with open("test_outputs.csv", "w") as f:
                for i in range(0, len(X)):
                    # looping through datapoints; i is datapoint
                    string = str(output[i][0])
                    if len(output[0]) > 1:
                        for j in range(1, len(output[i])):
                            # component wise
                            string += "," + str(output[i][j])
                    string += "\n"  # newline
                    f.write(string)

            print("Done, outputs saved to test_outputs.csv")

        try:
            threading.Thread(target=go).start()
        except Exception as e:
            print(e)

    def graph(self):
        def go():
            os.system("python3 plot_training.py")
        p = mp.Process(target=go).start()
        # p.join()

    def load_data(self):
        X = []
        Y = []
        print("Reading data from model_inputs.csv")
        with open("model_inputs.csv", "r") as f:
            next_line = str(f.readline())
            while next_line != "":
                line = next_line.split(",")
                this_x = []
                for i in line:
                    this_x.append(float(i))
                X.append(this_x)
                next_line = f.readline()

        print("Reading data from model_outputs.csv")
        with open("model_outputs.csv", "r") as f:
            next_line = str(f.readline())
            while next_line != "":
                line = next_line.split(",")
                this_y = []
                for i in line:
                    this_y.append(float(i))
                Y.append(this_y)
                next_line = f.readline()

        return (X, Y)

    def suicide(self):
        try:
            self.AI.suicide = True
            os.system("killall python3")
            exit()
        except Exception as e:
            print(e)
            pass

    def log_parameters(self):
        while True:
            try:
                p = copy.deepcopy(self.AI.spc)  # extract parame ters
                epoch = 0
                while True:
                    new_p = copy.deepcopy(self.AI.spc)

                    if new_p[0][0][0][0] != p[0][0][0][0]:
                        with open("parameter_log.csv", "a") as f:
                            f.write(str(epoch))
                            for i in new_p:
                                for j in i:
                                    for k in j:
                                        for l in k:
                                            f.write("," + str(l))
                            f.write("\n")
                            p = copy.deepcopy(new_p)
                            epoch += 1

            except Exception as e:
                with open("errorlog.txt", "a") as f:
                    f.write(str(e))

    def log_loss(self):
        while True:
            try:
                L = self.AI.bench
                epoch = 0
                while True:
                    new_L = self.AI.bench
                    if new_L != L:
                        #there has been an update
                        with open("history.csv", "a") as f:
                            f.write(str(epoch) + "," + str(L) + "\n")

                        epoch += 1
                        L = new_L
            except Exception as e:
                with open("errorlog.txt", "a") as f:
                    f.write(str(e))

    def test_drive(self):
        try:
            test_out = 0
            if self.model_type == "Katalina" or self.model_type == "Catherine" or self.model_type == "Lerochka":
                current_hyperparameters = self.AI.spc
                test_out = self.AI.forward_propagate(
                    current_hyperparameters, self.model_inputs)

            if self.model_type == "TurboKAN1" or self.model_type == "KAN_optimised":
                c = self.AI.spc
                w = self.AI.w
                test_out = []
                test_out = self.AI.forward_propagate(c, w, self.model_inputs)

            test_Y = []
            for i in test_out:
                test_Y.append(i[0])

            test_X = []
            for i in self.model_inputs:
                test_X.append(i[0])

            real_Y = []
            for i in self.model_outputs:
                real_Y.append(i[0])

            fig = plt.Figure()
            tests = plt.scatter(test_X, test_Y)
            real = plt.scatter(test_X, real_Y)
            plt.show()
        except Exception as e:
            print(e)


if __name__ == "__main__":
    os.system("rm history.csv")
    os.system("rm parameter_log.csv")
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec())
