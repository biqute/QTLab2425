import numpy as np
import cmath
import sys; sys.path.append("../utils")
from peak_width import peak_width
from read_metadata import read_metadata
import sys; sys.path.append("../classes")
from Fitter import Fitter
import math
from scipy import constants

# |S_21(f)|
# https://arxiv.org/pdf/1410.3365 eq. (2)
def model_modulus_resonance(f, a, b, c, d, A, phi, Q_i, Q_c, f_r):
    Q_l = 1/(1/Q_c + 1/Q_i)
    df = f - f_r
    y = a + b*df + c*df**2 + d*df**3 + A*np.abs(
            Q_l * cmath.exp(1j*phi) / np.abs(Q_c) / (
            1 + 2j * Q_l * df / f_r
        )
    )
    return y

basename = "empty_cavity\\40mK_1.0kHz_-45dBm"
data = np.loadtxt(f"..\\data\\{basename}.csv", delimiter=",")
metadata = read_metadata(f"..\\data\\{basename}_meta.csv")

fitter = Fitter()
fitter.datax = data[:, 0] # Hz
fitter.datay = np.sqrt(data[:, 1]**2 + data[:, 2]**2) # 1
fitter.sigmay = np.maximum(1e-5 + fitter.datax*0, np.abs(fitter.datay*0.01))
fitter.scaley = "dB" # "linear" (default), "log", "dB"
fitter.unity = "1"
fitter.scalex = lambda x: x / 1e9 # "linear" (default), "log", "dB"
fitter.unitx = "GHz"
fitter.title = f"Empty cavity resonance at {metadata["temperature"]} with power {metadata["power"]} and IF BW of {metadata["bandwidth"]}"
fitter.labelx = "Frequency"
fitter.labely = "$|S_{21}|$"
fitter.model = model_modulus_resonance
fitter.show_initial_model = False
fitter.show_plot = True
fitter.show_pvalue = False
fitter.file_name = f"..\\plots\\{basename}.pdf"
fitter.figure_size = (30, 24)

# [res1, res2, res3] = simultaneous_fitting([fitter1, ftter2, fitter3])
# fitter1.plot(res1)
# fitter2.plot(res2)
# fitter3.plot(res3)

Q_c = 10e3 # coupling quality factor
f_r = fitter.datax[np.argmax(fitter.datay)]
width = peak_width(fitter.datax, fitter.datay) # no - in front of datay
Q_i = f_r / width # internal quality factor TODO: NO THIS IS Q_l INSTEAD
Q_l = 1/(1/Q_c + 1/Q_i) # loaded quality factor
A = ( np.max(fitter.datay) - np.min(fitter.datay) ) * Q_c / Q_l
a = fitter.datay[0] - fitter.model(fitter.datax[0], 0.0, b = 0, c = 0, d = 0, A = A, phi = 0, Q_i = Q_i, Q_c = Q_c, f_r = f_r)

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
    "Q_i": (0, Q_i, None), 
    "Q_c": (0, Q_c, None), 
    "f_r": (0, f_r, None)
}
applied_dBm = float(metadata["power"][:-3]) - 45
applied_watt = 1e-3 * math.pow(10, applied_dBm / 10) # watt = 1mW * 10^(power/10), with power in dBm, TODO: check
Q_l_lambda = lambda params: 1/(1/params["Q_i"]["value"] + 1/params["Q_c"]["value"])
fitter.derived_params = {
    "Q_l": Q_l_lambda,
    "n": lambda params: 4 * constants.pi / constants.hbar / params["f_r"]["value"]**2 * (Q_l_lambda(params))**2 / params["Q_c"]["value"] * applied_watt, # https://arxiv.org/pdf/2006.04718 Eq. (20)
}
fitter.param_units = {
    "A":   "1", 
    "a":   "1", 
    "b":   "s", 
    "c":   "s^2", 
    "d":   "s^3", 
    "phi": "rad", 
    "f_r": "Hz",
    "Q_i": "1", 
    "Q_c": "1", 
    "Q_l": "1",
    "n": "1",
}
fitter.param_displayed_names = { 
    "phi": "\\phi", 
    "Q_l": "Q_\\ell",
    "n": "\\langle n \\rangle", 
    "A": "", 
    "a": "", 
    "b": "", 
    "c": "", 
    "d": "", 
    "phi": ""
}


res = fitter.plot_fit()