# Catherine.py  Model with activation functions that are well defined and fast to compute. May not be as versatile as full splines but nevertheless should be good enough.

# single KAN but multithreaded and optimised

# KAN network

import math
import numpy as np
import random
import splines
import copy
import multiprocessing as mp
import threading
import time


class NN:

    def __init__(self, structure, order=3, grids=10, learning_rate=0.0001, train_inputs=None, train_outputs=None):
        # trainable params
        self.grids = grids
        self.structure = structure
        self.layers = len(structure)-1
        self.pause = False

        self.train_inputs = train_inputs
        self.train_outputs = train_outputs
        self.order = order
        self.dw = 0.00001
        self.dc = 0.00001
        self.learning_rate = learning_rate
        # self.knots = np.linspace(0, 1, grids)

        # construct the the net by initilising its nodes and edges
        self.nodes = []
        for l in range(0, self.layers+1):
            self.nodes.append(np.zeros(structure[l]))

        self.spc = []
        self.edges = []
        # w[l][j][k]
        for l in range(0, self.layers):
            this_layer = []
            for j in range(0, structure[l]):
                col = []
                for k in range(0, structure[l+1]):
                    col.append(np.random.normal(0, 0.1, size=(5)))

                this_layer.append(col)

            self.spc.append(this_layer)
            self.edges.append(np.zeros((structure[l], structure[l+1])))

        self.dc = 0.000001
        self.dw = 0.000001
        # start logging

        #threading.Thread(target=self.injection).start()

        # output from activation

        # or, this can be simplified by focusing on one forward node at a time, gathering outputs from previous edges and nodes

        # can either store activation functions in matrices, or only care about the params
        # either ways we need a 5D object, 3 are for identification of position within the net, 2 for naviagating the free params
        # spc[l][j][k][g][n]
        # g is gridpoint, n

        # Here, w and fc need to become arrays to hold the individual values.
        # self.w = random.random()
        # self.fc, self.knots = splines.initialise_splines(3, grids)
        # we can go for different grid sizes at each edge, resulting in different knot counts. Or we can go for all the same.
        # having all the same grids saves calculation, so let's do that

        # need to generate a collection of free coefficients for each activation edge

        # now we need to initialise the activation functions, i.e. splines
        # we need to store the free coefficients and total coefficients


    def log(self):
        time.sleep(1)
        e = 0
        while True:
            try:
                with open("history.csv", "a") as history:
                    history.write(str(e) + "," + str(self.bench) + "\n")
                e = e+1


            except AttributeError:
                print("waiting")
                time.sleep(0.2)

            except Exception as e:
                print("Failed to log, reason: ", e)

            time.sleep(0.2)

    def spline(self, x, coefficients):
        # alles = sp.fill_coefficients(free_coefficients, self.knots)
        S = 0
        for i in range(0, 4):
            S += coefficients[i] * x ** i
        return (S)

    def silu(self, x):
        # this is the basis function
        try:
            return (x / (1 + np.exp(-x)))
        except OverflowError:
            print("overflow in silu")
            return (0)

    def activation(self, x, l, j, k, hyperparameters):
        # we want to use a specified local weight and spline coefficients so that we can observe effects of perturbation
        p = hyperparameters[l][j][k][4] * self.silu(x
                                                    ) + self.spline(x, hyperparameters[l][j][k])

        return (p)

    def run(self, hyperparameters, train_inputs):
        # first we must separate each data point
        out = []

        for this_input in train_inputs:
            # we must construct the network according to the specifications
            # nodes: 2D array following the structure
            nodes = []
            for j in self.structure:
                nodes.append(np.zeros(j))

            edges = copy.deepcopy(self.edges)

            # now copy the input vector to the first set of nodes
            nodes[0] = this_input

            # compute the output of the previous edges, then push to the next x
            for l in range(0, self.layers):
                # now we are looking at 2 neighbouring layers with internal indices k and j.
                # j is previous layer and k is next layer
                for k in range(0, self.structure[l+1]):
                    for j in range(0, self.structure[l]):
                        # computing the output of connecting edges
                        edges[l][j][k] = self.activation(
                            nodes[l][j], l, j, k, hyperparameters)
                # now go through each node in the next layer
                for k in range(0, self.structure[l+1]):
                    # sum up contributions from edges leading to this node
                    for j in range(0, self.structure[l]):
                        nodes[l+1][k] += edges[l][j][k]

                # now we need to normalise this layer, unless it's the last one'

            out.append(nodes[-1])

        return (out)

    def forward_propagate(self, hyperparameters, input_vectors):
        batch_size = len(input_vectors)
        # first layer
        # batch_out = []
        # total = []
        # # compute just the next layer
        # for i in range(0, len(input_vectors)):
        #     # now we are looking at each input vector.
        #     # find the edges
        #     # edges = np.zeros((self.structure[l], self.structure[l+1]))
        #     edges = []  # 2D rectangular matrix connecting between 2 layers
        #     for k in range(0, self.structure[l+1]):
        #         next_node_k = []  # paths leading to the kth node on the next row
        #         for j in range(0, self.structure[l]):
        #             # call the activation function
        #             this_edge = self.activation(
        #                 self.input_vectors[i][j], l, j, k, coefficients, weights)
        #             next_node_k.append(this_edge)
        #         edges.append(next_node_k)
        #
        #     # now we have all the edges computed, to get the next layer just sum up
        #     this_out = []
        #     for k in range(0, self.structure[l+1]):
        #         this_node = sum(edges[k])
        #         this_out.append(this_node)
        #         total.append(this_node)
        #     batch_out.append(this_out)
        #
        # # now the next layer has been calculated. We want to normalise it
        # # firstly find the maximum and minimum
        # biggest = max(total)
        # smallest = min(total)
        # # apply normalisation function`
        # for i in range(0, len(batch_out)):
        #     for j in range(0, len(batch_out[i])):
        #         batch_out[i][j] = (batch_out[i][j] -
        #                            smallest) / (biggest - smallest)
        #
        # input_vectors = copy.deepcopy(batch_out)

        # hidden layers
        batch_out = []

        for l in range(0, self.layers):
            batch_out = []
            # compute just the next layer
            for i in range(0, batch_size):
                # now we are looking at each input vector.
                # find the edges
                # edges = np.zeros((self.structure[l], self.structure[l+1]))
                edges = []  # 2D rectangular matrix connecting between 2 layers
                for k in range(0, self.structure[l+1]):
                    next_node_k = []  # paths leading to the kth node on the next row
                    for j in range(0, self.structure[l]):
                        # call the activation function
                        this_edge = self.activation(
                            input_vectors[i][j], l, j, k, hyperparameters)
                        next_node_k.append(this_edge)
                    edges.append(next_node_k)

                # now we have all the edges computed, to get the next layer just sum up
                this_out = []
                for k in range(0, self.structure[l+1]):
                    this_out.append(sum(edges[k]))
                batch_out.append(this_out)

            # now we want to change the scale of the grids
            # self.knots = np.linspace(min(total), max(total), self.grids)

            input_vectors = copy.deepcopy(batch_out)

        return (batch_out)

    def loss(self, hyperparameters):
        error = 0
        result = self.forward_propagate(hyperparameters, self.train_inputs)
        # print(result)
        for i in range(0, len(result)):
            for j in range(0, len(result[i])):
                error += (result[i][j] - self.train_outputs[i][j])**2
        # diff = self.train_outputs - self.run(c, w, self.train_inputs)
        # error = np.dot(diff, diff)
        return (error)

    def SPC_gradient(self, l, j, k, n):
        up = copy.deepcopy(self.spc)
        up[l][j][k][n] += self.dc
        down = copy.deepcopy(self.spc)
        down[l][j][k][n] -= self.dc
        gradient = (self.loss(up) -
                    self.loss(down)) / (2 * self.dc)

        return ([gradient, l, j, k, n])

    def modify_spc(self, results):
        self.spc[results[1]][results[2]][results[3]
                                         ][results[4]] += -self.learning_rate * results[0]

    def backpropagate(self):
        pool = mp.Pool()
        # we compute the gradient for each parameter and use stochastic gradient descent
        for l in range(0, self.layers):
            for j in range(0, self.structure[l]):
                for k in range(0, self.structure[l+1]):
                    # now for all of the c parameters (polynomial coefficients)
                    for n in range(0, 5):
                        # for g in range(0, self.grids):
                        pool.apply_async(self.SPC_gradient,
                                         args=(l, j, k, n), callback=self.modify_spc)

        pool.close()
        pool.join()

    def train(self, tolerance=0.01, preload_hyperparameters=None):

        if preload_hyperparameters != None:
            self.spc = preload_hyperparameters
        self.bench = self.loss(self.spc)
        self.backpropagate()
        new = self.loss(self.spc)
        improvement = new - self.bench
        self.bench = new
        self.epoch = 1
        prev_lr = self.learning_rate
        if improvement > 0:
            self.learning_rate = self.learning_rate * 1.01
        threading.Thread(target=self.log).start()

        #threading.Thread(target=self.manager).start()

        while self.pause == False:
            self.backpropagate()
            print("Loss: ", self.bench, "  |  Epoch: ", self.epoch)
            new = self.loss(self.spc)
            # we have advanced forwards in epochs, so we can make updates to learning_rate
            new_improvement = new - self.bench
            if new >= self.bench:
                #stayed same or got worse:
                self.learning_rate = self.learning_rate / 1.3
            else:

                if new_improvement > improvement:
                    #we are on the right trajectory
                    if self.learning_rate < prev_lr:
                        #if we increased the learning rate and got better performance, we do it again
                        self.learning_rate = self.learning_rate * 0.98
                    elif self.learning_rate > prev_lr:
                        self.learning_rate = self.learning_rate * 1.01
                if new_improvement < improvement:
                    if self.learning_rate < prev_lr:
                        self.learning_rate = self.learning_rate * 1.01
                    elif self.learning_rate > prev_lr:
                        self.learning_rate = self.learning_rate * 0.97
                #self.learning_rate += self.learning_rate**3 * (new_improvement - improvement)/ (self.learning_rate - prev_lr)


            #update our bench to new
            self.bench = new
            improvement = new_improvement
            prev_lr = self.learning_rate

            self.epoch += 1

        return(self.spc)



