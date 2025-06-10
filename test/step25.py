if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph

def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z

def f(x):
    y = x ** 2
    return y

x = Variable(np.array(1.0))
y = f(x) + 3

y.backward()

x.name = 'x'
y.name = 'y'

plot_dot_graph(y, verbose=False, to_file='f.png')
print(x.grad)