import numpy as np
import cmath
import sys; sys.path.append("../utils")
from peak_width import peak_width
from read_metadata import read_metadata
from linear_sampling import linear_sampling
from simultaneous_fit import simultaneous_fit
import sys; sys.path.append("../classes")
from Fitter import Fitter
import math
from scipy import constants

# ============================ UTILITIES ============================

# Characterization of superconducting resonant RF cavities for axion search with the QUAX experiment (Alessio Rettaroli Master Thesis)
# eq. (2.54)
def model_rettaroli_S11(f, f0, QL, k1, k2, A):
    delta = f/f0 - f0/f
    B = (k1-1-k2) / (1+k1+k2) # ex sqrtG
    return A * (B + 1j*QL*delta) / (1 + 1j * QL * delta)

# eq. (2.57)
def model_rettaroli_S21(f, f0, QL, k1, k2, A):
    delta = f/f0 - f0/f
    C = 2 * math.sqrt(k1*k2) / (1+k1+k2) # ex sqrtI
    return A * C / (1 + 1j*QL*delta)

def width_to_indeces(width, center, datax):  
    deltaf = ( datax[-1] - datax[0] ) / len(datax)
    min_f = center - width / 2.0
    max_f = center + width / 2.0
    min_f_idx = int((min_f - datax[0]) / deltaf)
    max_f_idx = int((max_f - datax[0]) / deltaf)
    min_f_idx = min(len(datax) - 1, max(0, min_f_idx))
    max_f_idx = min(len(datax) - 1, max(0, max_f_idx))

    if min_f_idx == max_f_idx:
        raise ValueError(f"min_index = max_index = {min_f_idx}, no data in this range")
    if min_f_idx > max_f_idx:
        raise ValueError("min_index > max_index, something went wrong")

    return (min_f_idx, max_f_idx)

def estimate_parameters(freqs, S21, S11):
    f0 = freqs[np.argmax(np.abs(S21))]
    width = peak_width(freqs, np.abs(S21)) # no - in front of datay

    QL = f0 / width # loaded quality factor

    theta21 = np.angle(S21)[np.argmax(np.abs(S21))] # Arg[S21(w_0)]
    theta11_up_to_pi = np.angle(S11)[np.argmax(np.abs(S21))] # Arg[S11(w_0)] up to pi due to sign of B

    (id_phi, _) = width_to_indeces(width / 3, f0, freqs)
    id_phi = np.argmax(np.abs(S21)) - max(1, np.argmax(np.abs(S21)) - id_phi) # we move at least one datapoint to the left
    phi = np.angle(S11[id_phi] / np.exp(1j * theta11_up_to_pi))
    f_phi = freqs[id_phi]
    d_phi = f_phi/f0 - f0/f_phi
    B = (1 - np.tan(phi)*d_phi*QL) / (1 + np.tan(phi)/d_phi/QL)

    theta11 = theta11_up_to_pi
    if B < 0: theta11 = np.angle(np.exp(1j * (theta11_up_to_pi - np.pi))) # remove pi and make sure you are in [-pi, pi]

    S21_resonance = np.max(np.abs(S21))
    S11_resonance = np.abs(S11)[np.argmax(np.abs(S21))]
    
    A = S11_resonance / np.abs(B)
    C = S21_resonance / A

    alpha = (1 - B)/(1 + B)
    k1 = 4 / (4*alpha - (C*(alpha + 1))**2)
    k2 = alpha*k1 - 1

    return {
        "f0": f0,
        "QL": QL,
        "k1": k1,
        "k2": k2,
        "A": A,
        "B": B,
        "C": C,
        "theta11": theta11,
        "theta21": theta21,
        "width": width
    }

# ============================ DATA LOADING ============================

basename = f"empty_cavity_3PC/65mK_1.0kHz_-5dBm"
dataS11 = np.loadtxt(f"../data/{basename}_S11.csv", delimiter=",")
dataS21 = np.loadtxt(f"../data/{basename}_S21.csv", delimiter=",")
freqs = dataS11[:,0] # Hz
S11 = dataS11[:,1] + 1j*dataS11[:,2]
S21 = dataS21[:,1] + 1j*dataS21[:,2]

# ============================ PARAMETER ESTIMATION ============================

guess = estimate_parameters(freqs, S21, S11)
f0 = guess["f0"]
QL = guess["QL"]
k1 = guess["k1"]
k2 = guess["k2"]
A = guess["A"]
width = guess["width"]
theta11 = guess["theta11"]
theta21 = guess["theta21"]

# import pprint
# pprint.pprint(guess)

# ============================ REAL PARTS FITTERS ============================

crop = width_to_indeces(5 * width, f0, freqs)

f21_real = Fitter()
f21_real.datax = freqs[crop[0]:crop[1]] # Hz
f21_real.datay = np.real(S21[crop[0]:crop[1]]) # 1
f21_real.sigmay = f21_real.datay * 0 + 1e-3
f21_real.model = lambda x, f0, QL, k1, k2, A, theta21: np.real(np.exp(1j*theta21)*model_rettaroli_S21(x, f0, QL, k1, k2, A))

f11_real = Fitter()
f11_real.datax = freqs[crop[0]:crop[1]] # Hz
f11_real.datay = np.real(S11[crop[0]:crop[1]]) # 1
f11_real.sigmay = f11_real.datay * 0 + 1e-3
f11_real.model = lambda x, f0, QL, k1, k2, A, theta11: np.real(np.exp(1j*theta11)*model_rettaroli_S11(x, f0, QL, k1, k2, A))


# ============================ PREPARING FOR FITTING ============================

metadata_S21 = read_metadata(f"../data/{basename}_S21_meta.csv")
f21_real.scaley = "linear" # "linear" (default), "log", "dB"
f21_real.unity = "1"
f21_real.scalex = lambda x: x / 1e9 # "linear" (default), "log", "dB"
f21_real.unitx = "GHz"
f21_real.title = f"Empty cavity S21 at {metadata_S21["temperature"]} with power {metadata_S21["power"]} and IF BW of {metadata_S21["bandwidth"]}"
f21_real.labelx = "Frequency"
f21_real.labely = "$\\text{Re}(S_{21})$"
f21_real.show_initial_model = True
f21_real.show_plot = True
f21_real.show_pvalue = False
f21_real.show_model = True
f21_real.file_name = f"../plots/{basename}_S21_real.pdf"
f21_real.figure_size = (30, 24)

metadata_S11 = read_metadata(f"../data/{basename}_S11_meta.csv")
f11_real.scaley = "linear" # "linear" (default), "log", "dB"
f11_real.unity = "1"
f11_real.scalex = lambda x: x / 1e9 # "linear" (default), "log", "dB"
f11_real.unitx = "GHz"
f11_real.title = f"Empty cavity S11 at {metadata_S11["temperature"]} with power {metadata_S11["power"]} and IF BW of {metadata_S11["bandwidth"]}"
f11_real.labelx = "Frequency"
f11_real.labely = "$\\text{Re}(S_{11})$"
f11_real.show_initial_model = True
f11_real.show_plot = True
f11_real.show_pvalue = False
f11_real.show_model = True
f11_real.file_name = f"../plots/{basename}_S11_real.pdf"
f11_real.figure_size = (30, 24)

f21_real.params = {
    "QL": (0, QL, None), 
    "f0": (0, f0, None),
    "k1": (0, k1, 100),
    "k2": (0, k2, 100),
    "A": (0, A, 100),
    "theta21": (-np.pi, theta21, np.pi),
}
f21_real.param_units = { "QL": "1", "Q0": "1", "f0": "Hz", "k1": "1", "k2": "1", "A": "1", "theta21": "1" }
f21_real.derived_params = {
    "Q0": lambda par: par["QL"]["value"] * (1 + par["k1"]["value"] + par["k2"]["value"]),
}
f21_real.param_displayed_names = { 
    "Q0": "Q_i = (1 + k_1 + k_2) Q_L",
    "QL": "Q_L",
    "f0": "f_r", 
    "k1": "\\kappa_1",
    "k2": "\\kappa_2",
    "theta21": "\\theta_{21}",
}
f11_real.params = {
    "QL": f21_real.params["QL"], 
    "f0": f21_real.params["f0"],
    "k1": f21_real.params["k1"],
    "k2": f21_real.params["k2"],
    "A": f21_real.params["A"],
    "theta11": (-np.pi, theta11, np.pi), # TODO: should not need to subtract pi
}
f11_real.param_units = { "QL": "1", "f0": "Hz", "k1": "1", "k2": "1", "A": "1", "theta11": "1" }
f11_real.param_displayed_names = { 
    "Q0": "Q_i",
    "QL": "Q_L",
    "f0": "f_r", 
    "k1": "\\kappa_1",
    "k2": "\\kappa_2",
    "theta11": "\\theta_{11}",
}

f21_imag = f21_real.deep_copy()
f21_imag.datay = np.imag(S21)[crop[0]:crop[1]] # 1
f21_imag.labely = "$\\text{Im}(S_{21})$"
f21_imag.model = lambda x, f0, QL, k1, k2, A, theta21: np.imag(np.exp(1j*theta21)*model_rettaroli_S21(x, f0, QL, k1, k2, A))
f21_imag.file_name = f"../plots/{basename}_S21_imag.pdf"

f11_imag = f11_real.deep_copy()
f11_imag.datay = np.imag(S11)[crop[0]:crop[1]] # 1
f11_imag.labely = "$\\text{Im}(S_{11})$"
f11_imag.model = lambda x, f0, QL, k1, k2, A, theta11: np.imag(np.exp(1j*theta11)*model_rettaroli_S11(x, f0, QL, k1, k2, A))
f11_imag.file_name = f"../plots/{basename}_S11_imag.pdf"

# ============================ FITTING ============================

res = simultaneous_fit([f21_real, f21_imag, f11_real, f11_imag])
f21_real.plot(res)
f21_imag.plot(res)
f11_real.plot(res)
f11_imag.plot(res)