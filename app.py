import sys

from PyQt5.QtWidgets import (
    QApplication, QDialog, QMainWindow, QMessageBox
)
from PyQt5.uic import loadUi
from PyQt5 import QtCore, QtGui, QtWidgets



class Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
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

        self.label_2 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.Layout.addWidget(self.label_2, 1, 0, 1, 1)

        self.Start_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.Start_button.setObjectName("Start_button")
        self.Start_button.clicked.connect(self.start_training)
        self.Layout.addWidget(self.Start_button, 5, 0, 1, 1)

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
        self.label_5 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_5.setObjectName("label_5")
        self.Layout.addWidget(self.label_5, 3, 0, 1, 1)

        self.Pause_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.Pause_button.setObjectName("pushButton")
        self.Pause_button.clicked.connect(self.pause)
        self.Layout.addWidget(self.Pause_button, 6, 0, 1, 1)

        self.label_4 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_4.setObjectName("label_4")
        self.Layout.addWidget(self.label_4, 4, 0, 1, 1)

        self.label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label.setObjectName("label")
        self.Layout.addWidget(self.label, 2, 0, 1, 1)

        self.label_3 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.Layout.addWidget(self.label_3, 0, 0, 1, 1)

        self.Shape_entry = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.Shape_entry.setObjectName("Shape_entry")
        self.Layout.addWidget(self.Shape_entry, 1, 2, 1, 1)

        self.LossTolerance_entry = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.LossTolerance_entry.setObjectName("LossTolerance_entry")
        self.LossTolerance_entry.setText("0.01")
        self.Layout.addWidget(self.LossTolerance_entry, 3, 2, 1, 1)

        self.Batchsize_entry = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.Batchsize_entry.setObjectName("Batchsize_entry")
        self.Layout.addWidget(self.Batchsize_entry, 4, 2, 1, 1)

        self.Resume_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.Resume_button.setObjectName("Resume_button")
        self.Resume_button.clicked.connect(self.resume)
        self.Layout.addWidget(self.Resume_button, 6, 2, 1, 1)

        self.Run_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.Run_button.setObjectName("Run_button")
        self.Run_button.clicked.connect(self.run)
        self.Layout.addWidget(self.Run_button, 5, 2, 1, 1)

        self.Testdrive_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.Testdrive_button.setObjectName("Testdrive_button")
        self.Layout.addWidget(self.Testdrive_button, 7, 0, 1, 1)
        self.Plot_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.Plot_button.setObjectName("Plot_button")
        self.Layout.addWidget(self.Plot_button, 7, 2, 1, 1)

        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.Layout.addItem(spacerItem1, 8, 3, 1, 1)
        self.Layout.addItem(spacerItem, 8, 0, 1, 1)

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
        self.LossTolerance_entry.setToolTip(_translate("Frame", "The acceptable loss. The network stops training when this loss has been reached."))
        self.label.setText(_translate("Frame", "Initial learning rate"))
        self.label_3.setText(_translate("Frame", "Select Model"))
        self.Resume_button.setText(_translate("Frame", "Resume Training"))
        self.Run_button.setText(_translate("Frame", "Run"))
        self.Shape_entry.setToolTip(_translate("Frame", "Example: 4, 2, 1"))
        self.label_2.setToolTip(_translate("Frame", "The first number is dimension of input vector, last is dimension of output vector. Middle numbers correspond to size of hidden layers."))
        self.comboBox.setToolTip(_translate("Frame", "Select different network designs. KAN_optimised and turboKAN are made according to the paper; Catherine is single grid cubic spline; Katalina does split minibatch training."))
        self.Testdrive_button.setText(_translate("Frame", "Testdrive"))
        self.Plot_button.setText(_translate("Frame", "Plot Training Progress"))

    def start_training(self):
        #firstly let's extract the user inputs
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

        #learning rate and loss tolerance are optional
        lr = str(self.LearningRate_entry.text())
        if lr != "":
            lr = float(lr)
        else:
            lr = 0.001 #default value

        loss_tolerance = self.LossTolerance_entry.text()
        if loss_tolerance != "":
            loss_tolerance = float(loss_tolerance)
        else:
            loss_tolerance = 0.01

        model_type = str(self.comboBox.currentText())

        self.model_inputs, self.model_outputs = self.load_data()

        if model_type == "Katalina":
            print("Selected model: Katalina")
            from network_models import katalina
            self.AI = katalina.NN(structure, learning_rate=lr,
                                  train_inputs=model_inputs, train_outputs=model_outputs)
            print("Initialisation complete, now training...")
            self.trained_hyperparameters = self.AI.train()

        elif model_type == "TurboKAN1":
            print("selected model: TurboKAN1")

        elif model_type == "KAN_optimised":
            print("selected model: KAN_optimised")

        elif model_type == "Catherine":
            print("Selected model: Catherine")

        else:
            print("Not implemented yet!")

    def resume(self):
        self.AI.pause = False
        self.AI.train(preload_hyperparameters=self.trained_hyperparameters)


    def pause(self):
        self.AI.pause = True

    def run(self):
        self.AI.forward_propagate(self.trained_hyperparameters, self.model_inputs, self.model_outputs)

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

        return(X, Y)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec())
