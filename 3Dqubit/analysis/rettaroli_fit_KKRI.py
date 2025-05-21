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
    min_f_idx = min(len(datax) - 1, max(0, min_f_idx))
    max_f_idx = min(len(datax) - 1, max(0, max_f_idx))

    if min_f_idx == max_f_idx:
        raise ValueError(f"min_index = max_index = {min_f_idx}, no data in this range")
    if min_f_idx > max_f_idx:
        raise ValueError("min_index > max_index, something went wrong")

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
S11_complex = data["S11"]["real"] + 1j*data["S11"]["imag"]
S21_complex = data["S21"]["real"] + 1j*data["S21"]["imag"]
S11_abs = np.abs(S11_complex)
S21_abs = np.abs(S21_complex)

# Fitter for S21
metadata_S21 = read_metadata(f"../data/{basename}_S21_meta.csv")
f21_real = Fitter()
f21_real.datax = datax # Hz
f21_real.datay = data["S21"]["real"] # 1
f21_real.sigmay = f21_real.datay * 0 + 1e-3
f21_real.scaley = "linear" # "linear" (default), "log", "dB"
f21_real.unity = "1"
f21_real.scalex = lambda x: x / 1e9 # "linear" (default), "log", "dB"
f21_real.unitx = "GHz"
f21_real.title = f"Empty cavity S21 at {metadata_S21["temperature"]} with power {metadata_S21["power"]} and IF BW of {metadata_S21["bandwidth"]}"
f21_real.labelx = "Frequency"
f21_real.labely = "$\\text{Re}(S_{21})$"
f21_real.model = lambda x, f0, QL, k1, k2, A, theta21: np.real(np.exp(1j*theta21)*model_rettaroli_S21(x, f0, QL, k1, k2, A))
f21_real.show_initial_model = True
f21_real.show_plot = True
f21_real.show_pvalue = False
f21_real.show_model = True
f21_real.file_name = f"../plots/{basename}_S21_real.pdf"
f21_real.figure_size = (30, 24)

# Fitter for S11
metadata_S11 = read_metadata(f"../data/{basename}_S11_meta.csv")
f11_real = Fitter()
f11_real.datax = datax # Hz
f11_real.datay = data["S11"]["imag"] # 1
f11_real.sigmay = f11_real.datay * 0 + 1e-3
f11_real.scaley = "linear" # "linear" (default), "log", "dB"
f11_real.unity = "1"
f11_real.scalex = lambda x: x / 1e9 # "linear" (default), "log", "dB"
f11_real.unitx = "GHz"
f11_real.title = f"Empty cavity S11 at {metadata_S11["temperature"]} with power {metadata_S11["power"]} and IF BW of {metadata_S11["bandwidth"]}"
f11_real.labelx = "Frequency"
f11_real.labely = "$\\text{Re}(S_{11})$"
f11_real.model = lambda x, f0, QL, k1, k2, A, theta11: np.real(np.exp(1j*theta11)*model_rettaroli_S11(x, f0, QL, k1, k2, A))
f11_real.show_initial_model = True
f11_real.show_plot = True
f11_real.show_pvalue = False
f11_real.show_model = True
f11_real.file_name = f"../plots/{basename}_S11_real.pdf"
f11_real.figure_size = (30, 24)

# Parameter estimation
f_r = datax[np.argmax(S21_abs)]
width = peak_width(datax, S21_abs) # no - in front of datay
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

(crop_min_idx, crop_max_idx) = width_to_indeces(5 * width, f_r, f21_real.datax)
f21_real.datax = f21_real.datax[crop_min_idx:crop_max_idx]
f21_real.datay = f21_real.datay[crop_min_idx:crop_max_idx]
f21_real.sigmay = f21_real.sigmay[crop_min_idx:crop_max_idx]
f11_real.datax = f11_real.datax[crop_min_idx:crop_max_idx]
f11_real.datay = f11_real.datay[crop_min_idx:crop_max_idx]
f11_real.sigmay = f11_real.sigmay[crop_min_idx:crop_max_idx]
S11_complex = S11_complex[crop_min_idx:crop_max_idx]
S21_complex = S21_complex[crop_min_idx:crop_max_idx]
S11_abs = S11_abs[crop_min_idx:crop_max_idx]
S21_abs = S21_abs[crop_min_idx:crop_max_idx]

############ Parameter estimation ############

a = res_back["params"]["a"]["value"]
S11_omega = S11_abs[np.argmax(S21_abs)]

# (id1, id2) = width_to_indeces(width, f_r, f21_real.datax)
# S11_delta_omega = f21_real.datax[id2] - a # (f21_real.datax[id2] + f21_real.datax[id1]) / 2.0 - a 
# G = 1 / (2*S11_delta_omega/S11_omega - 1)

theta21 = np.angle(S21_complex)[np.argmax(S21_abs)] # Arg[S21(w_0)]
theta11 = np.angle(S11_complex)[np.argmax(S21_abs)] # Arg[S11(w_0)]
aaaa = np.angle(S11_complex / np.exp(1j * theta11))
idx1 = np.min(np.arange(0, len(aaaa))[aaaa > 0])
idx2 = np.max(np.arange(0, len(aaaa))[aaaa < 0])
f1 = linear_sampling(np.array([np.pi/4]), aaaa[idx1:(idx2+1)], f11_real.datax[idx1:(idx2+1)])[0]
delta1 = f1/f_r - f_r/f1
# G = (delta1 * QL * (1 - delta1) / (1 + delta1 * QL))**2 # WRONG CALCULATION!!
G = (delta1 * QL * (1 - delta1*QL) / (1 + delta1*QL))**2

for n in np.arange(1, 20)/2:
    phi = np.pi / n
    f_phi = linear_sampling(np.array([phi]), aaaa[idx1:(idx2+1)], f11_real.datax[idx1:(idx2+1)])[0]
    delta_phi = f_phi/f_r - f_r/f_phi
    G_phi = (delta_phi * QL * (np.tan(phi) - delta_phi*QL) / (np.tan(phi) + delta_phi*QL))**2
    print(f"φ = π/{n} => G_φ = {G_phi}")

A = S11_omega / G
alpha = (1 - np.sqrt(G))/(1 + np.sqrt(G))
I = (np.max(S21_abs) / A)**2
k1 = 4 / (4*alpha - I*(alpha + 1)**2)
k2 = alpha*k1 - 1
# k1 = 1 / (2 - np.sqrt(I))
# k2 = k1

print(f"G = {G} = {((k1-k2-1)/(k1+k2+1))**2}")
print(f"I = {I} = {4*k1*k2/(k1+k2+1)**2}")

################################################

f21_real.params = {
    "QL": (0, QL, None), 
    "f0": (0, f_r, None),
    "k1": (0, k1, 100),
    "k2": (0, k2, 100),
    "A": (0, A, 100),
    "theta21": (-np.pi, theta21, np.pi),
#    "a": (res_back["params"]["a"]["value"] * 0.8, res_back["params"]["a"]["value"], res_back["params"]["a"]["value"] * 1.2),
#    "b": (None, res_back["params"]["b"]["value"], None),
#    "c": (None, res_back["params"]["c"]["value"], None),
}
print(f21_real.params)
f21_real.param_units = {
    "QL": "1",
    "Q0": "1",
    "f0": "Hz",
    "k1": "1",
    "k2": "1",
    "A": "1",
    "a": "1",
    "b": "Hz^{-1}",
    "c": "Hz^{-2}",
    "theta21": "1",
}
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
    "theta11": (-np.pi, theta11, np.pi),
}
f11_real.param_units = {
    "QL": "1",
    "f0": "Hz",
    "k1": "1",
    "k2": "1",
    "A": "1",
    "theta11": "1",
}
f11_real.param_displayed_names = { 
    "Q0": "Q_i",
    "QL": "Q_L",
    "f0": "f_r", 
    "k1": "\\kappa_1",
    "k2": "\\kappa_2",
    "theta11": "\\theta_{11}",
}

f21_imag = f21_real.deep_copy()
f21_imag.datay = data["S21"]["imag"][crop_min_idx:crop_max_idx] # 1
f21_imag.labely = "$\\text{Im}(S_{21})$"
f21_imag.model = lambda x, f0, QL, k1, k2, A, theta21: np.imag(np.exp(1j*theta21)*model_rettaroli_S21(x, f0, QL, k1, k2, A))
f21_imag.file_name = f"../plots/{basename}_S21_imag.pdf"

f11_imag = f11_real.deep_copy()
f11_imag.datay = data["S11"]["imag"][crop_min_idx:crop_max_idx] # 1
f11_imag.labely = "$\\text{Im}(S_{11})$"
f11_imag.model = lambda x, f0, QL, k1, k2, A, theta11: np.imag(np.exp(1j*theta11)*model_rettaroli_S11(x, f0, QL, k1, k2, A))
f11_imag.file_name = f"../plots/{basename}_S11_imag.pdf"

# res = simultaneous_fit([f21_real, f21_imag, f11_real, f11_imag])
# f21_real.plot(res["results"][0])
# f21_imag.plot(res["results"][1])
# f11_real.plot(res["results"][2])
# f11_imag.plot(res["results"][3])

f11_real.plot_fit()

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