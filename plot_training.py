#loss function graph

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
import math
import random
def run():
    def readfile():
        file = open("history.csv")

        tL = []
        T =[]
        epoch = 1
        next_line = file.readline()
        while next_line != "":
            row = next_line.split(",")
            try:
                tL.append(math.log(float(row[1])+1))
                T.append(math.log(epoch+1))
                epoch += 1
                next_line = file.readline()
            except Exception as error:
                print(error)
                break

        #print("finished reading data")
        return(T, tL)




    # initial data
    x, y = readfile()

    # creating the first plot and frame
    fig, ax = plt.subplots()
    global graph
    colour = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
    graph = ax.plot(x,y,color=colour)[0]

    plt.xlabel("ln( epochs )")
    plt.ylabel("ln(Total Loss (MAE))")


    # updates the data and graph
    def update(frame):
        x, y = readfile()
        graph = ax.plot(x,y,color=colour)[0]

        # creating a new graph or updating the graph
        graph.set_xdata(x)
        graph.set_ydata(y)
        plt.xlim(x[0], x[-1])
        plt.ylim(0,max(y))

    anim = FuncAnimation(fig, update, frames = None)
    plt.show()
run()
