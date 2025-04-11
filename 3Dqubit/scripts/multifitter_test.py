import sys; sys.path.append("../classes")
from Fitter import Fitter
import sys; sys.path.append("../utils")
from simultaneous_fit import simultaneous_fit
import numpy as np

def model1(x, a, b):
    return a + b * x
def model2(x, a, c):
    return a + c * np.pow(x, 2)

f0 = Fitter()
f0.datax = np.linspace(0, 10, 100)
f0.datay = model1(f0.datax, 1, 2) + np.random.normal(0, 0.5, len(f0.datax))
f0.sigmay = np.ones(len(f0.datax)) * 0.5
f0.params = {"a": (None, 1, None), "b": (None, 1, None)}
f0.model = model1
f0.derived_params = {"a+b": lambda params: params["a"]["value"] + params["b"]["value"]}
f0.param_units = {"a": "1", "b": "1", "a+b": "1"}
f0.show_pvalue = False
f0.show_plot = True

f1 = Fitter()
f1.datax = np.linspace(0, 10, 100)
f1.datay = model2(f1.datax, 1, 3) + np.random.normal(0, 0.5, len(f1.datax))
f1.sigmay = np.ones(len(f1.datax)) * 0.5
f1.params = {"a": (None, 1, None), "c": (None, 1, None)}
f1.model = model2
f1.param_units = {"a": "1", "c": "1"}
f1.show_pvalue = False
f1.show_plot = True

res = simultaneous_fit([f0, f1])

print(res)

f0.plot(res["results"][0])
f1.plot(res["results"][1])