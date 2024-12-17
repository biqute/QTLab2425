import numpy as np
import cmath
import sys; sys.path.append("../classes")
from Fitter import Fitter
import math


fitter = Fitter()
fitter.datax = np.array([0,1,2,3])
fitter.datay = np.array([0,1,2,3])
fitter.sigmay = np.array([0.1,0.1, 0.1, 0.08])
fitter.model = lambda datax, a, b: a + b*datax 
fitter.number_of_errorbars = 3
fitter.show_initial_model = True
fitter.params = {
    "a":   (0.0, 0.25, None),
    "b":   (None, 0.1, None),
}
fitter.param_units = {
    "a":   "1", 
    "b":   "s", 
}
fitter.unitx = "Hz"
fitter.unity = "1"

fitter.plot_fit()