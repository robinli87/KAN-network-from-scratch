#KAN network

import math
import numpy as np
import random
import splines
import copy

class NN:

    def __init__(self, order, grids=10, learning_rate=0.001, train_inputs=None, train_outputs=None):
        #trainable params
        self.grids = grids
        self.w = random.random()
        self.fc, self.knots = splines.initialise_splines(3, grids)
        self.train_inputs = train_inputs
        self.train_outputs = train_outputs
        self.order = order
        self.dw = 0.00001
        self.dc = 0.00001
        self.learning_rate = learning_rate

        #now we need to initialise the activation functions, i.e. splines
        #we need to store the free coefficients and total coefficients


    def spline(self, x, free_coefficients):
        alles = splines.fill_coefficients(free_coefficients, self.knots)
        S = splines.spline(x, alles, self.knots)
        return(S)

    def silu(self, x):
        return(x / (1 + np.exp(x)))

    def phi(self,x, c, w):
        p = w * (self.silu(x) + self.spline(x, c))
        return(p)

    def run(self, c, w, train_inputs):
        #element separation
        batch_outs = []
        for i in range(0, len(train_inputs)):
            #layer 1
            this_result = self.phi(train_inputs[i], c, w)

            batch_outs.append(this_result)

        return(np.array(batch_outs))

    def loss(self, c, w):
        diff = self.train_outputs - self.run(c, w, self.train_inputs)
        error = np.dot(diff, diff)
        return(error)

    def backpropagate(self):
        #we compute the gradient for each parameter and use stochastic gradient descent
        upper = self.loss(self.fc, self.w+self.dw)
        lower = self.loss(self.fc, self.w-self.dw)
        gradient = (upper - lower)/(self.dw*2)
        self.dw += -self.learning_rate * gradient

        #now for all of the c parameters (polynomial coefficients)
        for i in range(0, len(self.fc)):
            for j in range(0, len(self.fc[i])):
                up = copy.deepcopy(self.fc)
                low = copy.deepcopy(self.fc)
                up[i][j] += self.dc
                low[i][j] -= self.dc
                #print(up[i][j])
                #print(low[i][j])
                gradient = (self.loss(up, self.w) - self.loss(low, self.w))/(2 * self.dc)

                self.fc[i][j] += -self.learning_rate * gradient


    def train(self):
        bench = self.loss(self.fc, self.w)
        self.backpropagate()
        new = self.loss(self.fc, self.w)
        epoch = 1

        while (new < bench) or (epoch < 10000):
            if (new >= bench):
                self.learning_rate = self.learning_rate / 2
                print(self.learning_rate)
            else:
                self.learning_rate = self.learning_rate * 1.005
            print("Loss: ", new)
            bench = new
            self.backpropagate()
            new = self.loss(self.fc, self.w)
            epoch += 1

            if epoch > 1000:
                break #quit if we are taking too long

        return(self.fc, self.w)


