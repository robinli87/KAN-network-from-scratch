# single KAN but multithreaded and optimised

# KAN network

import math
import numpy as np
import random
import splines
import copy
import multiprocessing as mp
import threading

sp = splines.spline_tools()


class NN:

    def __init__(self, structure, order=3, grids=10, learning_rate=0.001, train_inputs=None, train_outputs=None):
        # trainable params
        self.grids = grids
        self.structure = structure
        self.layers = len(structure)-1

        self.train_inputs = train_inputs
        self.train_outputs = train_outputs
        self.order = order
        self.dw = 0.00001
        self.dc = 0.00001
        self.learning_rate = learning_rate
        self.knots = np.linspace(0, 1, grids)

        # construct the the net by initilising its nodes and edges
        self.nodes = []
        for l in range(0, self.layers+1):
            self.nodes.append(np.zeros(structure[l]))

        self.spc = []  # spline coefficients, 4D or 5D object
        for l in range(0, self.layers):
            this_layer = []
            for j in range(0, self.structure[l]):
                last_col = []
                for k in range(0, self.structure[l+1]):
                    last_col.append(sp.initialise_splines(order, grids))
                this_layer.append(last_col)
            self.spc.append(this_layer)

        self.w = []  # weights, size equal to the number of edges
        self.edges = []
        # w[l][j][k]
        for l in range(0, self.layers):
            self.w.append(np.random.normal(
                0, 1, size=(structure[l], structure[l+1])))
            self.edges.append(np.zeros((structure[l], structure[l+1])))

        self.dc = 0.000001
        self.dw = 0.000001

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

    def spline(self, x, free_coefficients):
        # alles = sp.fill_coefficients(free_coefficients, self.knots)
        S = sp.spline(x, free_coefficients, self.knots)
        return (S)

    def silu(self, x):
        # this is the basis function
        return (x / (1 + np.exp(x)))

    def activation(self, x, l, j, k, these_FC, W):
        # we want to use a specified local weight and spline coefficients so that we can observe effects of perturbation
        p = W[l][j][k] * self.silu(x) + self.spline(x, these_FC[l][j][k])

        return (p)

    def normalise_linear(self, vector):

        N = []
        for i in vector:
            N.append((float(i) - min(vector)) /
                     (max(vector) - min(vector)))
        return (N)

    def run(self, coefficients, weights, train_inputs):
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
                            nodes[l][j], l, j, k, coefficients, weights)
                # now go through each node in the next layer
                for k in range(0, self.structure[l+1]):
                    # sum up contributions from edges leading to this node
                    for j in range(0, self.structure[l]):
                        nodes[l+1][k] += edges[l][j][k]

                # now we need to normalise this layer, unless it's the last one'
                if l != self.layers - 1:
                    nodes[l+1] = self.normalise_linear(nodes[l+1])
                    print(nodes[l+1])

            out.append(nodes[-1])

        return (out)

    def forward_propagate(self, coefficients, weights, input_vectors):
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

        for l in range(0, self.layers-1):
            batch_out = []
            total = []
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
                            input_vectors[i][j], l, j, k, coefficients, weights)
                        next_node_k.append(this_edge)
                    edges.append(next_node_k)

                # now we have all the edges computed, to get the next layer just sum up
                this_out = []
                for k in range(0, self.structure[l+1]):
                    this_node = sum(edges[k])
                    this_out.append(this_node)
                    total.append(this_node)
                batch_out.append(this_out)

            # now the next layer has been calculated. We want to normalise it
            # firstly find the maximum and minimum
            biggest = max(total)
            smallest = min(total)
            # apply normalisation function`
            for i in range(0, batch_size):
                for j in range(0, len(batch_out[i])):
                    batch_out[i][j] = (batch_out[i][j] -
                                       smallest) / (biggest - smallest)

            input_vectors = copy.deepcopy(batch_out)

        # now we are at the penultimate layer. The last layer does not require normalisation

        batch_out = []
        # compute just the next layer
        for i in range(0, batch_size):
            # now we are looking at each input vector.
            # find the edges
            # edges = np.zeros((self.structure[l], self.structure[l+1]))
            edges = []  # 2D rectangular matrix connecting between 2 layers
            for k in range(0, self.structure[-1]):
                next_node_k = []  # paths leading to the kth node on the next row
                for j in range(0, self.structure[-2]):
                    # call the activation function
                    this_edge = self.activation(
                        input_vectors[i][j], self.layers-1, j, k, coefficients, weights)
                    next_node_k.append(this_edge)
                edges.append(next_node_k)

            # now we have all the edges computed, to get the next layer just sum up
            this_out = []
            for k in range(0, self.structure[-1]):
                this_node = sum(edges[k])
                this_out.append(this_node)
                # total.append(this_node)
            batch_out.append(this_out)

        return (batch_out)

    def loss(self, c, w):
        error = 0
        result = self.forward_propagate(c, w, self.train_inputs)
        for i in range(0, len(result)):
            for j in range(0, len(result[i])):
                error += (result[i][j] - self.train_outputs[i][j])**2
        # diff = self.train_outputs - self.run(c, w, self.train_inputs)
        # error = np.dot(diff, diff)
        return (error)

    def weights_gradient(self, l, j, k):
        # this is multiprocessed. Unfortunately, we can't make changes to global variables in these local processes'
        upper = copy.deepcopy(self.w)
        lower = copy.deepcopy(self.w)
        upper[l][j][k] += dw
        lower[l][j][k] -= dw
        upper_loss = self.loss(self.spc, upper)
        lower_loss = self.loss(self.spc, lower)
        gradient = (upper_loss - lower_loss)/(self.dw*2)

        return ([gradient, l, j, k])

    def modify_weights(self, results):
        # here we can make changes to global variables
        self.w[results[1]][results[2]][results[3]
                                       ] -= self.learning_rate * results[0]

    def SPC_gradient(self, l, j, k, g, n):
        up = copy.deepcopy(self.spc)
        up[l][j][k][g][n] += self.dc
        down = copy.deepcopy(self.spc)
        down[l][j][k][g][n] -= self.dc
        gradient = (self.loss(up, self.w) -
                    self.loss(down, self.w)) / (2 * self.dc)

        return ([gradient, l, j, k, g, n])

    def modify_spc(self, results):
        self.spc[results[1]][results[2]][results[3]][results[4]
                                                     ][results[5]] += -self.learning_rate * results[0]

    def backpropagate(self):
        pool = mp.Pool()
        # we compute the gradient for each parameter and use stochastic gradient descent
        for l in range(0, self.layers):
            for j in range(0, self.structure[l]):
                for k in range(0, self.structure[l+1]):
                    pool.apply_async(self.weights_gradient, args=(
                        l, j, k), callback=self.modify_weights)

                    # now for all of the c parameters (polynomial coefficients)

                    for n in range(0, self.order):
                        for g in range(0, self.grids):
                            pool.apply_async(self.SPC_gradient,
                                             args=(l, j, k, n, g), callback=self.modify_spc)

        pool.close()
        pool.join()

    def train(self):
        bench = self.loss(self.spc, self.w)
        self.backpropagate()
        new = self.loss(self.spc, self.w)
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
            new = self.loss(self.spc, self.w)
            epoch += 1

            if epoch > 1000:
                break  # quit if we are taking too long

        return (self.spc, self.w)
