#Data generator - generates random training data. Currently using the function specified by the paper.

#f(x1, x2, x3, x4) = exp(sin(x1**2 + x2**2) + sin(x3**2 + x4**2))

import random
import math
import numpy as np

batch_size = 400
X = []
Y = []
# for i in range(0, batch_size):
#     x1 = np.linspace(0.9, 1.4, 500)
#     X.append(x1)
#     this_output = 10 * x1 ** -6 + 20 * x1 ** -12
#     Y.append(this_output)

X = np.linspace(0.9, 1.5, batch_size)
Y = X ** -2

with open("model_inputs.csv", "w") as inputfile:
    string = ""
    for i in range(0, len(X)):
        #looping through datapoints; i is datapoint

        string = str(X[i])
        string += "\n"  #newline
        inputfile.write(string)

with open("model_outputs.csv", "w") as outfile:
    string = ""
    for i in range(0, len(Y)):
        #looping through datapoints; i is datapoint

        string = str(Y[i])
        string += "\n"  #newline
        outfile.write(string)


X = []
for i in range(0, batch_size):
    x1 = random.gauss(0, 2)
    X.append([x1])
with open("test_inputs.csv", "w") as inputfile:
    for i in range(0, len(X)):
        #looping through datapoints; i is datapoint
        string = str(X[i][0])
        if len(X[0]) > 1:
            for j in range(1, len(X[i])):
                #component wise
                string += "," + str(X[i][j])
        string += "\n"  #newline
        inputfile.write(string)

print("done")
