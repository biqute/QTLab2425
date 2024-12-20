import numpy as np
import cmath
import sys; sys.path.append("../classes")
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

data = np.loadtxt("..\\data\\resonances\\500mK_-20dBm.txt")

fitter = Fitter()
fitter.datax = data[:, 0]
fitter.datay = np.power(10 + fitter.datax*0, data[:, 1]/20)
fitter.sigmay = np.maximum(1e-5 + fitter.datax*0, np.abs(fitter.datay*0.01))
fitter.scaley = "dB" # "linear" (default), "log", "dB"
fitter.scalex = "linear" # "linear" (default), "log", "dB"
fitter.model = model_modulus_notch
fitter.show_initial_model = False

Q_c = 2e3 # coupling quality factor
f_r = fitter.datax[np.argmin(fitter.datay)]

# estimating full width half max FWHM
half_height_value = np.max(fitter.datay) - (np.max(fitter.datay) - np.min(fitter.datay)) / math.sqrt(2)
hits = []
above = fitter.datay[0] > half_height_value
for i in range(1, len(fitter.datay)):
    new_above = fitter.datay[i] > half_height_value
    if new_above != above: 
        hits.append((fitter.datax[i] + fitter.datax[i-1]) / 2)
        above = new_above

Q_i = f_r / (hits[-1] - hits[0]) # internal quality factor
Q_l = 1/(1/Q_c + 1/Q_i) # loaded quality factor

A = 1.0
a = fitter.datay[0] - fitter.model(fitter.datax[0], 0.0, b = 0, c = 0, d = 0, A = A, phi = 0, Q_l = Q_l, Q_c = Q_c, f_r = f_r)

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