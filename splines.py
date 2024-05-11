import numpy as np
import random


class spline_tools:

    def __init__(self):
        #put some default values for some of the parameters:
        self.gridpoints = 10
        self.highest_power = 3

    def initialise_splines(self, N, G):
        self.highest_power = N
        self.N = N +1
        self.gridpoints = G

        #make first segment's coefficients corresponding to t0->t1
        free_coefficients = [np.random.normal(0, 1, size=(self.N))]
        #consider the next row
        #firstly let's get the free coefficients in:
        #3 are dependent, so do N-2

        #knots = np.linspace(0, 1, G)
        #knot generation can be done somewhere else if this initialise_spline is to be called too many times

        #this block is responsible for adding one knot
        for i in range(1, G):
            next_free = np.random.normal(0, 1, size=(N-2))
            free_coefficients.append(next_free)

        return(free_coefficients)

            #we need to add G-1 knotslen(free_coefficients)-1

    def next_dependents(self, a, free_b, t):
        #want to calculate b2 b1 b0 for the next row
        N = self.N

        # b = [] #just using range as a placeholder because of sticky array problem
        # for i in range(0, 3):
        #     b.append(0) #placeholder for b2 b1 b0
        #
        # for i in range(0, len(free_b)):
        #     b.append(free_b[i])
        #y the first 3 terms need to be modified
        #print(b)
        b = np.concatenate((np.zeros(3), free_b))

        #this can be optimised by fixing the max power to an integer, e.g. x**3. The for loops and guessing N is really inefficient.
        #

        b2 = a[2]
        #D = a - b
        for j in range(3, N):
            b2 += 0.5 * j*(j-1)*(a[j]-b[j])*t**(j-2)
        b[2] = b2


        b1 = a[1]
        for j in range(2, N):
            b1 += j * (a[j] - b[j]) * t ** (j-1)
        b[1] = b1


        b0 = a[0]
        for j in range(1, N):
            b0 += (a[j] - b[j]) * t**j
        b[0] = b0
        #print(b)

        return(b)

    def fill_coefficients(self, free_coefficients, knots, selected_knot):
        all_coefficients = [free_coefficients[0]]
        #next layer
        for i in range(0, selected_knot):
            all_coefficients.append(self.next_dependents(all_coefficients[i],
                                                         free_coefficients[i+1], knots[i+1]))

        return(all_coefficients)

    def B(self, x, C):
        #C is the row of coefficients for this basis
        #now we can compute the output of our polynomial
        #need to separate into basis functions
        y = 0
        for i in range(0, self.N):
            y += C[i] * x ** i
        return(y)

    #now we want to define the spline function, given the coefficients
    def spline(self, x, free_coefficients, knots):
        #this ingests all coefficients and outputs blindly; doesn't care which one was free and which one wasn't
        #we require 0<x<1
        #firstly we must determine which interval x is in
        #spline selector
        k = 0

        while (k+1 < self.gridpoints):
            if x >= knots[k+1]:
                k = k + 1
            else:
                break
        #print(k)
        #now we retrieve the right set of coefficients

        #we can save computation by only calculating coefficients up to the required knot
        coefficients = self.fill_coefficients(free_coefficients, knots, k)


        #compute B usiyng this set of coefficients
        y = self.B(x, coefficients[-1])
        return(y)






