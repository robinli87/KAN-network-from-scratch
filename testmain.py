# shortmain.py

import KAN_optimised as K
import random


AI = K.NN([1, 3, 3, 1])

# print("--------------------------initialised free coefficients------------------------")
spline_coefficients = AI.spc
# print(len(spline_coefficients))

# print("--------------------------weights-----------------------")
weights = AI.w
# print(len(weights))

# print(len(AI.nodes))
# prepare input data
train_inputs = []
for i in range(0, 10):
    train_inputs.append([random.random()])

output = AI.run(spline_coefficients, weights, train_inputs)
print(output)
