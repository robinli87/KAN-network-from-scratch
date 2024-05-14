# shortmain.py

import katalina as K
import random
import math
import matplotlib.pyplot as plt


structure = [4, 2, 1, 1]

batch_size = 100

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

AI = K.NN(structure, train_inputs=X, train_outputs=Y)

# print("--------------------------initialised free coefficients------------------------")
spline_coefficients = AI.spc
# print(len(spline_coefficients))

# print("--------------------------weights-----------------------")
# weights = AI.w
# print(len(weights))

# print(len(AI.nodes))
# prepare input data

trained_spc = AI.train(tolerance=0.02, sub_batch_size=20)

# test_outs = AI.forward_propagate(trained_spc, X)
#
# print(test_outs)
#
# fig = plt.Figure()
#
# XX = []
# YP = []
# for i in range(0, len(test_outs)):
#     YP.append(test_outs[i][0])
#     XX.append(X[i][0])
#
# plt.plot(XX, YP)
# plt.plot(XX, Y)
#
# plt.show()
# L = AI.loss(spline_coefficients, weights)
# print(L)
