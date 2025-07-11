import numpy as np
import cryofit as cr
from scipy.signal import find_peaks
import math

"""
f0 = 7436392514.0
QL = 1000
theta21 = 0
theta11 = 0
A = 1
k1 = 1
k2 = 2
# B = -0.5
# Qi = 4000

freqs = np.linspace(f0 - 500e6, f0 + 500e6, num=int(1e4))
S11 = cr.model_S11(freqs, f0, QL, k1, k2, A, theta11)
S21 = cr.model_S21(freqs, f0, QL, k1, k2, A, theta21)
S11 += np.random.normal(0,1e-3,len(freqs))
S21 += np.random.normal(0,1e-3,len(freqs))

# import matplotlib.pyplot as plt
# plt.plot(freqs, np.abs(S21))
# plt.scatter(freqs, np.abs(S21))
# plt.show()

est = cr.estimate_parameters(freqs, S11, S21)
print("f0 =", est["f0"])
print("A =", est["A"])
print("B =", est["B"])
print("C =", est["C"])
print("k1 =", est["k1"])
print("k2 =", est["k2"])
print("Qi =", est["Qi"])
print("QL =", est["QL"])
print("-----")
"""

##### REAL DATA

# basename = f"empty_cavity_3PC/65mK_1.0kHz_-5dBm"
basename = f"qubit_hot_3PC/run_fixed_power/300000mK_1.0kHz_-5dBm_span_250000000.0"
dataS11 = np.loadtxt(f"../data/{basename}_S11.csv", delimiter=",")
dataS21 = np.loadtxt(f"../data/{basename}_S21.csv", delimiter=",")
freqs = dataS11[:,0] # Hz
S11 = dataS11[:,1] + 1j*dataS11[:,2]
S21 = dataS21[:,1] + 1j*dataS21[:,2]

# basename = f"akash_dixit_S11_S21/akash"
# dataS11 = np.load(f"../data/{basename}_S11.npz")
# dataS21 = np.load(f"../data/{basename}_S21.npz")
# freqs = dataS11["freqs"] # Hz
# S11 = dataS11["data_I"] + 1j*dataS11["data_Q"]
# S21 = dataS21["data_I"] + 1j*dataS21["data_Q"]

df = (freqs[-1] - freqs[0]) / len(freqs)
peaks, _ = find_peaks(np.abs(S21), height=np.abs(S21)[0]*100, distance=500e6/df)
peaks = [np.argmax(np.abs(S21))]

for id in peaks:
    idx_min = id - math.floor(50e6/df)
    idx_max = id + math.floor(50e6/df)
    res = cr.fit_resonance(freqs[idx_min:idx_max], S11[idx_min:idx_max], S21[idx_min:idx_max], 
        plot_S11_real = False,
        plot_S11_imag = False,
        plot_S11_abs = False,
        plot_S11_phase = False,
        plot_S21_real = False,
        plot_S21_imag = False,
        plot_S21_abs = True,
        plot_S21_phase = False,
    )