#Data generator - generates random training data. Currently using the function specified by the paper.

#f(x1, x2, x3, x4) = exp(sin(x1**2 + x2**2) + sin(x3**2 + x4**2))

import random
import math

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

with open("model_inputs.csv", "w") as inputfile:
    for i in range(0, len(X)):
        #looping through datapoints; i is datapoint
        string = str(X[i][0])
        if len(X[0]) > 1:
            for j in range(1, len(X[i])):
                #component wise
                string += "," + str(X[i][j])
        string += "\n"  #newline
        inputfile.write(string)

with open("model_outputs.csv", "w") as outfile:
    for i in range(0, len(Y)):
        #looping through datapoints; i is datapoint
        string = str(Y[i][0])
        if len(Y[0]) > 1:
            for j in range(1, len(Y[i])):
                #component wise
                string += "," + str(Y[i][j])
        string += "\n"  #newline
        outfile.write(string)

print("done")
