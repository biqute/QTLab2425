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

# Characterization of superconducting resonant RF cavities for axion search with the QUAX experiment (Alessio Rettaroli Master Thesis)
# eq. (2.54)
def model_rettaroli_S11(f, f0, QL, k1, k2, A):
    delta = f/f0 - f0/f
    return A * ((k1-1-k2) / (1+k1+k2) + 1j*QL*delta) / (1 + 1j * QL * delta)

# eq. (2.57)
def model_rettaroli_S21(f, f0, QL, k1, k2, A):
    delta = f/f0 - f0/f
    return A * 2 * math.sqrt(k1*k2) / (1+k1+k2) / (1 + 1j*QL*delta)

def width_to_indeces(width, center, datax):  
    deltaf = ( datax[-1] - datax[0] ) / len(datax)
    min_f = center - width / 2.0
    max_f = center + width / 2.0
    min_f_idx = int((min_f - datax[0]) / deltaf)
    max_f_idx = int((max_f - datax[0]) / deltaf)
    min_f_idx = max(0, min_f_idx)
    max_f_idx = min(len(datax) - 1, max_f_idx)
    return (min_f_idx, max_f_idx)

#basename = f"empty_cavity_S11_S21_S22/40mK_1.0kHz_-10dBm"
basename = f"empty_cavity_3PC/65mK_1.0kHz_-5dBm"
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
f21.model = lambda x, f0, QL, k1, k2, A, a, b, c: np.abs(model_rettaroli_S21(x, f0, QL, k1, k2, A)) + a + b * (x-f0) + c * (x-f0)**2
f21.show_initial_model = False
f21.show_plot = True
f21.show_pvalue = False
f21.show_model = True
f21.file_name = f"../plots/{basename}_S21.pdf"
f21.figure_size = (30, 24)

# Fitter for S11
metadata_S11 = read_metadata(f"../data/{basename}_S11_meta.csv")
f11 = Fitter()
f11.datax = datax # Hz
f11.datay = np.sqrt(data["S11"]["real"]**2 + data["S11"]["imag"]**2) # 1
f11.sigmay = f11.datay * 0 + 1e-3
f11.scaley = "dB" # "linear" (default), "log", "dB"
f11.unity = "1"
f11.scalex = lambda x: x / 1e9 # "linear" (default), "log", "dB"
f11.unitx = "GHz"
f11.title = f"Empty cavity S11 at {metadata_S11["temperature"]} with power {metadata_S11["power"]} and IF BW of {metadata_S11["bandwidth"]}"
f11.labelx = "Frequency"
f11.labely = "$|S_{11}|$"
f11.model = lambda x, f0, QL, k1, k2, A: np.abs(model_rettaroli_S11(x, f0, QL, k1, k2, A))
f11.show_initial_model = False
f11.show_plot = True
f11.show_pvalue = False
f11.show_model = True
f11.file_name = f"../plots/{basename}_S11.pdf"
f11.figure_size = (30, 24)

# Parameter estimation
f_r = f21.datax[np.argmax(f21.datay)]
width = peak_width(f21.datax, f21.datay) # no - in front of datay
QL = f_r / width # loaded quality factor

# Fitter background
back_data = np.loadtxt(f"../data/{basename}_S21_wide.csv", delimiter=",")
back_datax = back_data[:,0]
back_data = {"real": back_data[:,1], "imag": back_data[:,2]}
back_datay = np.sqrt(back_data["real"]**2 + back_data["imag"]**2)
back_sigmay = back_datay * 0 + 1e-6

(crop_min_idx, crop_max_idx) = width_to_indeces(150 * width, f_r, back_datax)

fback = Fitter()
fback.datax = np.concatenate((back_datax[0:crop_min_idx], back_datax[crop_max_idx:-1]))
fback.datay = np.concatenate((back_datay[0:crop_min_idx], back_datay[crop_max_idx:-1]))
fback.sigmay = np.concatenate((back_sigmay[0:crop_min_idx], back_sigmay[crop_max_idx:-1]))
fback.scaley = "linear" # "linear" (default), "log", "dB"
fback.unity = "1"
fback.scalex = lambda x: x / 1e9 # "linear" (default), "log", "dB"
fback.unitx = "GHz"
fback.title = f"Background S21"
fback.labelx = "Frequency"
fback.labely = ""
fback.model = lambda x, a, b, c: a + b * (x-f_r) + c * (x-f_r)**2
fback.show_initial_model = False
fback.show_plot = True
fback.show_pvalue = False
fback.show_model = True
fback.figure_size = (30, 24)
fback.params = {
    "a": (0, 1, None), 
    "b": (None, 0, None),
    "c": (None, 0, None),
}
fback.param_units = {
    "a": "1",
    "b": "Hz^{-1}",
    "c": "Hz^{-2}",
}
res_back = fback.fit()

# Parameter estimation
a = res_back["params"]["a"]["value"]
(id1, id2) = width_to_indeces(width, f_r, f11.datax)
S11_omega = f11.datay[np.argmax(f21.datay)]
S11_delta_omega = f11.datax[id2] - a  # (f11.datax[id2] + f11.datax[id1]) / 2.0 - a 
G = 1 / (2*S11_delta_omega/S11_omega - 1)
A = S11_omega / G
alpha = (1 - np.sqrt(G))/(1 + np.sqrt(G))
I = (np.max(f21.datay) / A)**2
k1 = 4 / (4*alpha - I*(alpha + 1)**2)
k2 = alpha*k1 - 1 + 0.5 # TODO TODO TODO TODO TODO TODO TODO TODO TODO
# k1 = 1 / (2 - np.sqrt(I))
# k2 = k1

(crop_min_idx, crop_max_idx) = width_to_indeces(5 * width, f_r, f21.datax)
f21.datax = f21.datax[crop_min_idx:crop_max_idx]
f21.datay = f21.datay[crop_min_idx:crop_max_idx]
f21.sigmay = np.maximum(1e-5 + f21.datax*0, np.abs(f21.datay*0.01)) # TODO
f11.datax = f11.datax[crop_min_idx:crop_max_idx]
f11.datay = f11.datay[crop_min_idx:crop_max_idx]
f11.sigmay = np.maximum(1e-5 + f11.datax*0, np.abs(f11.datay*0.01)) # TODO

f21.params = {
    "QL": (0, QL, None), 
    "f0": (0, f_r, None),
    "k1": (0, k1, 100),
    "k2": (0, k2, 100),
    "A": (0, A, None),
    "a": (res_back["params"]["a"]["value"] * 0.8, res_back["params"]["a"]["value"], res_back["params"]["a"]["value"] * 1.2),
    "b": (None, res_back["params"]["b"]["value"], None),
    "c": (None, res_back["params"]["c"]["value"], None),
}
print(f21.params)
f21.param_units = {
    "QL": "1",
    "Q0": "1",
    "f0": "Hz",
    "k1": "1",
    "k2": "1",
    "A": "1",
    "a": "1",
    "b": "Hz^{-1}",
    "c": "Hz^{-2}",
}
f21.derived_params = {
    "Q0": lambda par: par["QL"]["value"] * (1 + par["k1"]["value"] + par["k2"]["value"]),
}
f21.param_displayed_names = { 
    "Q0": "Q_i = (1 + k_1 + k_2) Q_L",
    "QL": "Q_L",
    "f0": "f_r", 
    "k1": "\\kappa_1",
    "k2": "\\kappa_2",
}

f11.params = {
    "QL": (0, QL, None), 
    "f0": (0, f_r, None),
    "k1": f21.params["k1"],
    "k2": f21.params["k2"],
    "A": (0, A, None),
}
f11.param_units = {
    "QL": "1",
    "f0": "Hz",
    "k1": "1",
    "k2": "1",
    "A": "1",
}
f11.param_displayed_names = { 
    "Q0": "Q_i",
    "QL": "Q_L",
    "f0": "f_r", 
    "k1": "\\kappa_1",
    "k2": "\\kappa_2",
}

f11_phase = Fitter()
f11_phase.datax = np.copy(f11.datax) # Hz
f11_phase.datay = np.angle(np.exp(2j)*(data["S11"]["real"][crop_min_idx:crop_max_idx] + 1j*data["S11"]["imag"][crop_min_idx:crop_max_idx])) # rad
f11_phase.sigmay = f11_phase.datay * 0 + 1e-3
f11_phase.scaley = "linear" # "linear" (default), "log", "dB"
f11_phase.unity = "rad"
f11_phase.scalex = f11.scalex
f11_phase.unitx = f11.unitx
f11_phase.title = f11.title
f11_phase.labelx = f11.labelx
f11_phase.labely = "$\\text{arg}(S_{11})$"
f11_phase.model = lambda x, f0, QL, k1, k2, A, theta11: np.angle(np.exp(1j * theta11) * model_rettaroli_S11(x, f0, QL, k1, k2, A))
f11_phase.show_initial_model = f11.show_initial_model # = False
f11_phase.show_plot = f11.show_plot
f11_phase.show_pvalue = f11.show_pvalue
f11_phase.show_model = f11.show_model
f11_phase.file_name = f"../plots/{basename}_S11_phase.pdf"
f11_phase.figure_size = f11.figure_size
f11_phase.params = f11.params | {"theta11": (-np.pi, 0, +np.pi)}
f11_phase.param_units = f11.param_units | {"theta11": "rad"}
f11_phase.param_displayed_names = f11.param_displayed_names | {"theta11": "\\theta_{11}"}

f21_phase = Fitter()
f21_phase.datax = np.copy(f21.datax) # # Hz
f21_phase.datay = np.angle(np.exp(2j)*(data["S21"]["real"][crop_min_idx:crop_max_idx] + 1j*data["S21"]["imag"][crop_min_idx:crop_max_idx])) # rad
f21_phase.sigmay = f21_phase.datay * 0 + 1e-3
f21_phase.scaley = "linear" # "linear" (default), "log", "dB"
f21_phase.unity = "rad"
f21_phase.scalex = f21.scalex
f21_phase.unitx = f21.unitx
f21_phase.title = f21.title
f21_phase.labelx = f21.labelx
f21_phase.labely = "$\\text{arg}(S_{11})$"
f21_phase.model = lambda x, f0, QL, k1, k2, A, theta21: np.angle(np.exp(1j * theta21) * model_rettaroli_S11(x, f0, QL, k1, k2, A))
f21_phase.show_initial_model = f21.show_initial_model # = False
f21_phase.show_plot = f21.show_plot
f21_phase.show_pvalue = f21.show_pvalue
f21_phase.show_model = f21.show_model
f21_phase.file_name = f"../plots/{basename}_S21_phase.pdf"
f21_phase.figure_size = f21.figure_size
f21_phase.params = f21.params | {"theta21": (-np.pi, 0, +np.pi)}
f21_phase.params.pop("a")
f21_phase.params.pop("b")
f21_phase.params.pop("c")
f21_phase.param_units = f21.param_units | {"theta21": "rad"}
f21_phase.param_displayed_names = f21.param_displayed_names | {"theta21": "\\theta_{21}"}

res = simultaneous_fit([f21, f11, f11_phase, f21_phase])
f21.plot(res["results"][0])
f11.plot(res["results"][1])
f11_phase.plot(res["results"][2])
f21_phase.plot(res["results"][3])

# plot phase and model with matplotlib
# import matplotlib.pyplot as plt
# 
# def model_rettaroli_S11_phase(f, f0, QL, k1, k2, A):
#     delta = f/f0 - f0/f
#     return np.angle(((k1-1-k2) / (1+k1+k2) + 1j*QL*delta) / (1 + 1j * QL * delta))
# 
# plt.scatter(f11.datax, , label="data")
# plt.plot(f11.datax, model_rettaroli_S11_phase(f11.datax, 
#     res["results"][0]["params"]["f0"]["value"], 
#     res["results"][0]["params"]["QL"]["value"], 
#     res["results"][0]["params"]["k1"]["value"], 
#     res["results"][0]["params"]["k2"]["value"], 
#     res["results"][0]["params"]["A"]["value"]), 
# label="model")
# plt.title("Phase S11")
# plt.xlabel("f")
# plt.show()