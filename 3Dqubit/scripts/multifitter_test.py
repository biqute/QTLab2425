import sys; sys.path.append("../classes")
from Fitter import Fitter
import sys; sys.path.append("../utils")
from simultaneous_fit import simultaneous_fit
import numpy as np

def model1(x, a, b):
    return a + b * x
def model2(x, a, c):
    return a + c * x**2

f1 = Fitter()
f1.datax = np.linspace(0, 10, 100)
f1.datay = model1(f1.datax, 1, 2) + np.random.normal(0, 0.5, len(f1.datax))
f1.sigmay = np.ones(len(f1.datax)) * 0.5
f1.params = {"a": (None, 1, None), "b": (None, 1, None)}
f1.model = model1

f2 = Fitter()
f2.datax = np.linspace(0, 10, 100)
f2.datay = model2(f2.datax, 1, 3) + np.random.normal(0, 0.5, len(f2.datax))
f2.sigmay = np.ones(len(f2.datax)) * 0.5
f2.params = {"a": (None, 1, None), "c": (None, 1, None)}
f2.model = model2

res = simultaneous_fit([f1, f2])