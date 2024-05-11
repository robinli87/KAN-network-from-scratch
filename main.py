#prepare model data; let's do a sin curve between -90 degrees and +90 degrees but with some noise added
#let's go for 20 datapoints'
import KAN_optimised
import numpy as np
import math
import random
import matplotlib.pyplot as plt

model_size = 100
model_inputs = np.random.uniform(-1, 1, size=(model_size))
model_inputs = np.sort(model_inputs)
model_outputs = np.sin(model_inputs * math.pi) + np.random.normal(0, 0.1, size=(model_size))


order = 5
AI = KAN_optimised.NN(order, train_inputs=model_inputs, train_outputs=model_outputs)
c, w = AI.train()
print(c)
print(w)
print("---------------------------------------------")
#now we run this on test data

test_inputs = np.linspace(-1.2, 1.2, 100)
test_out = AI.run( c, w, test_inputs)
hidden_answer = np.sin(test_inputs * math.pi )

print(test_inputs)
print(test_out)

fig = plt.Figure()
plt.title("Output Results Plots")
plt.plot(test_inputs, test_out, color="r")
plt.plot(model_inputs, model_outputs, color="g")
#plt.plot(test_inputs, hidden_answer, color="c")
txt="""
Red line: learnt function.
Green line: Input data.
Cyan line: hidden trend in the sample"""
plt.figtext(1, 1, txt, wrap=True, horizontalalignment='center', fontsize=12)
plt.show()
