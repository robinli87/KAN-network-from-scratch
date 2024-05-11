import matplotlib.pyplot as plt
import numpy as np
import splines
import time

grids = 10

sp = splines.spline_tools()
frei = sp.initialise_splines(4, grids)
knots = np.linspace(0, 1, grids)

#try to plot this function and see if it is indeed smooth
fig = plt.Figure()
X = np.linspace(0, 1, 100)
Y=[]
start = time.time()
for x in X:
    Y.append(sp.spline(x, frei, knots))
Y = np.array(Y)
print(time.time() - start)
plt.plot(X, Y)
plt.show()
