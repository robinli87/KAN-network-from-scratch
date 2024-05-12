# shortmain.py

import catherine as K
import random
import math
import matplotlib.pyplot as plt


structure = [1, 2, 1]

batch_size = 20

X = []
Y = []
for i in range(0, batch_size):
    x = i / batch_size
    y = math.sin(x * math.pi) + random.random() * 0.1
    X.append([x])
    Y.append([y])

AI = K.NN(structure, train_inputs=X, train_outputs=Y)

# print("--------------------------initialised free coefficients------------------------")
spline_coefficients = AI.spc
# print(len(spline_coefficients))

# print("--------------------------weights-----------------------")
# weights = AI.w
# print(len(weights))

# print(len(AI.nodes))
# prepare input data

trained_spc = AI.train()

test_outs = AI.forward_propagate(trained_spc, X)

print(test_outs)

fig = plt.Figure()

XX = []
YP = []
for i in range(0, len(test_outs)):
    YP.append(test_outs[i][0])
    XX.append(X[i][0])

plt.plot(XX, YP)
plt.plot(XX, Y)

plt.show()
# L = AI.loss(spline_coefficients, weights)
# print(L)
