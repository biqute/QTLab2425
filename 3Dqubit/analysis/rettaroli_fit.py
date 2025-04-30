import numpy as np
import cmath
import sys; sys.path.append("../utils")
from peak_width import peak_width
from read_metadata import read_metadata
import sys; sys.path.append("../classes")
from Fitter import Fitter
import math
from scipy import constants

# Characterization of superconducting resonant RF cavities for axion search with the QUAX experiment (Alessio Rettaroli Master Thesis)
# eq. (2.54)
def model_rettaroli_S11(f, f0, Q0, k1, k2, A11):
    delta = f/f0 - f0/f
    x = 1 + k2 + 1j * Q0 * delta
    return A11*(k1 - x)/(k1 + x)

# eq. (2.57)
def model_rettaroli_S21(f, f0, Q0, k1, k2, A21):
    delta = f/f0 - f0/f
    return A21*(2 * math.sqrt(k1*k2))/(1 + k1 + k2 + 1j * Q0 * delta)    



#basename = f"empty_cavity_S11_S21_S22/40mK_1.0kHz_-10dBm"
basename = f"empty_cavity_3PC/300K_1.0kHz_-5dBm"
dataS11 = np.loadtxt(f"../data/{basename}_S11.csv", delimiter=",")
dataS21 = np.loadtxt(f"../data/{basename}_S21.csv", delimiter=",")
data = {
    "S11": {"real": dataS11[:,1], "imag": dataS11[:,2]},
    "S21": {"real": dataS21[:,1], "imag": dataS21[:,2]},
}
datax = dataS11[:,0] # Hz

# Fitter for S21
metadata_S21 = read_metadata(f"../data/{basename}_S21_meta.csv")
f21 = Fitter()
f21.datax = datax # Hz
f21.datay = np.sqrt(data["S21"]["real"]**2 + data["S21"]["imag"]**2) # 1
f21.sigmay = f21.datay * 0 + 1e-3
f21.scaley = "dB" # "linear" (default), "log", "dB"
f21.unity = "1"
f21.scalex = lambda x: x / 1e9 # "linear" (default), "log", "dB"
f21.unitx = "GHz"
f21.title = f"Empty cavity S21 at {metadata_S21["temperature"]} with power {metadata_S21["power"]} and IF BW of {metadata_S21["bandwidth"]}"
f21.labelx = "Frequency"
f21.labely = "$|S_{21}|$"
f21.model = model_rettaroli_S21
f21.show_initial_model = True
f21.show_plot = True
f21.show_pvalue = False
f21.show_model = True
f21.file_name = f"../plots/{basename}_S21.pdf"
f21.figure_size = (30, 24)

# Fitter for S11
metadata_S11 = read_metadata(f"../data/{basename}_S11_meta.csv")
f11 = Fitter()
f11.datax = datax # Hz
f11.datay = np.sqrt(data["S21"]["real"]**2 + data["S21"]["imag"]**2) # 1
f11.sigmay = f21.datay * 0 + 1e-3
f11.scaley = "dB" # "linear" (default), "log", "dB"
f11.unity = "1"
f11.scalex = lambda x: x / 1e9 # "linear" (default), "log", "dB"
f11.unitx = "GHz"
f11.title = f"Empty cavity S21 at {metadata_S11["temperature"]} with power {metadata_S11["power"]} and IF BW of {metadata_S11["bandwidth"]}"
f11.labelx = "Frequency"
f11.labely = "$|S_{11}|$"
f11.model = model_rettaroli_S11
f11.show_initial_model = True
f11.show_plot = True
f11.show_pvalue = False
f11.show_model = True
f11.file_name = f"../plots/{basename}_S11.pdf"
f11.figure_size = (30, 24)


# Parameter estimation
f_r = f21.datax[np.argmax(f21.datay)]
width = peak_width(f21.datax, f21.datay) # no - in front of datay
Q_i = f_r / width # internal quality factor

# Parameter estimation when we know A_11, A_21 = 1
# A = np.max(f21.datay) # A = S_(21) (w = w_r)
# B = 2*np.pi * np.max(f21.datax) * f21.datay[np.argmax(f21.datax)] # B = lim (w -> infty) [w * |S_(21) (w)|]
# a = np.sqrt(B * Q_i / 2 / (2 * np.pi * f_r))
# b = B/A * Q_i / (2 * np.pi * f_r) - 1
# k1 = 0.5 * (b + np.sqrt(b**2 - 4*a**2))
# k2 = 0.5 * (b - np.sqrt(b**2 - 4*a**2))

# Parameter estimation when we don't know A_11, A_21
A11 = f11.datay[-1]
b = f21.datax[-1] * f21.datay[-1] * Q_i / np.max(f21.datay) / (2 * np.pi * f_r) - 1
k1 = (1+b) / 2 * (np.min(f21.datay) / A11 + 1)  # TODO np.min(f21.datay) = S21(w_0), may fail! 
k2 = b - k1

print("k1:", k1, "k2:", k2)


deltaf = ( f21.datax[-1] - f21.datax[0] ) / len(f21.datax)
min_f = f_r - 2.5*width
max_f = f_r + 2.5*width
min_f_idx = int((min_f - f21.datax[0]) / deltaf)
max_f_idx = int((max_f - f21.datax[0]) / deltaf)
min_f_idx = max(0, min_f_idx)
max_f_idx = min(len(f21.datax) - 1, max_f_idx)

# f21.datax = f21.datax[min_f_idx:max_f_idx]
# f21.datay = f21.datay[min_f_idx:max_f_idx]
# f21.sigmay = np.maximum(1e-5 + f21.datax*0, np.abs(f21.datay*0.01)) # TODO

f21.params = {
    "Q0": (0, Q_i, None), 
    "f0": (0, f_r, None),
    "k1": (0, k1, None),
    "k2": (0, k2, None),
    "A11": (0, A11, None),
    "A21": (0, A21, None),
}
f21.param_units = {
    "Q0": "1",
    "f0": "Hz",
    "k1": "1",
    "k2": "1",
    "A11": "1",
    "A21": "1",
}
f21.param_displayed_names = { 
    "Q0": "Q_i",
    "f0": "f_r", 
    "k1": "\\kappa_1",
    "k2": "\\kappa_2",
    "A11": "A_{11}",
    "A21": "A_{21}"
}


res = f21.plot_fit()