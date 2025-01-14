import numpy as np
import cmath
import sys; sys.path.append("../classes")
import sys; sys.path.append("../utils")
from peak_width import peak_width
from Fitter import Fitter
import math

# |S_21(f)|
# https://arxiv.org/pdf/1410.3365 eq. (1)
def model_modulus_notch(f, a, b, c, d, A, phi, Q_l, Q_c, f_r):
    df = f - f_r
    y = a + b*df + c*df**2 + d*df**3 + A*np.abs(
        1 - Q_l * cmath.exp(1j*phi) / np.abs(Q_c) / (
            1 + 2j * Q_l * df / f_r
        )
    )
    return y

data = np.loadtxt("..\\data\\gap_run12_iq\\Q_res50.txt", delimiter=",")


fitter = Fitter()
fitter.datax = data[:, 0]
fitter.datay = np.sqrt(data[:, 1]**2 + data[:, 2]**2)
fitter.sigmay = np.maximum(1e-5 + fitter.datax*0, np.abs(fitter.datay*0.01))
fitter.scaley = "dB" # "linear" (default), "log", "dB"
fitter.scalex = "linear" # "linear" (default), "log", "dB"
fitter.model = model_modulus_notch
fitter.show_initial_model = True

Q_c = 50e3 # coupling quality factor
f_r = fitter.datax[np.argmin(fitter.datay)]
width = peak_width(fitter.datax, -fitter.datay)
Q_i = f_r / width # internal quality factor
Q_l = 1/(1/Q_c + 1/Q_i) # loaded quality factor
A = ( np.max(fitter.datay) - np.min(fitter.datay) ) * Q_c / Q_l
a = fitter.datay[0] - fitter.model(fitter.datax[0], 0.0, b = 0, c = 0, d = 0, A = A, phi = 0, Q_l = Q_l, Q_c = Q_c, f_r = f_r)

deltaf = ( fitter.datax[-1] - fitter.datax[0] ) / len(fitter.datax)
min_f = f_r - 2.5*width
max_f = f_r + 2.5*width
min_f_idx = int((min_f - fitter.datax[0]) / deltaf)
max_f_idx = int((max_f - fitter.datax[0]) / deltaf)
min_f_idx = max(0, min_f_idx)
max_f_idx = min(len(fitter.datax) - 1, max_f_idx)

fitter.datax = fitter.datax[min_f_idx:max_f_idx]
fitter.datay = fitter.datay[min_f_idx:max_f_idx]
fitter.sigmay = np.maximum(1e-5 + fitter.datax*0, np.abs(fitter.datay*0.01)) # TODO

fitter.params = {
    "A":   (0.0, A, None),
    "a":   (None, a, None),
    "b":   (None, 0.0, None),
    "c":   (None, 0.0, None),
    "d":   (None, 0.0, None),
    "phi": (-math.pi, 0, math.pi), 
    "Q_l": (0, Q_l, None), 
    "Q_c": (0, Q_c, None), 
    "f_r": (0, f_r, None)
}
fitter.param_units = {
    "A":   "1", 
    "a":   "1", 
    "b":   "s", 
    "c":   "s^2", 
    "d":   "s^3", 
    "phi": "rad", 
    "Q_l": "1", 
    "Q_c": "1", 
    "f_r": "Hz"
}
fitter.param_displayed_names = { "phi": "\\phi" }
fitter.unitx = "Hz"
fitter.unity = "1"
fitter.title = "Resonance"

res = fitter.plot_fit()
res["plot"].show()