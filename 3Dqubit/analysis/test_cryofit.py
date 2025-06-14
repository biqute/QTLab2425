import numpy as np
import cryofit as cr
from scipy.signal import find_peaks
import math

#basename = f"empty_cavity_3PC/65mK_1.0kHz_-5dBm"
#dataS11 = np.loadtxt(f"../data/{basename}_S11.csv", delimiter=",")
#dataS21 = np.loadtxt(f"../data/{basename}_S21.csv", delimiter=",")
# freqs = dataS11[:,0] # Hz
# S11 = dataS11[:,1] + 1j*dataS11[:,2]
# S21 = dataS21[:,1] + 1j*dataS21[:,2]

basename = f"akash_dixit_S11_S21/akash"
dataS11 = np.load(f"../data/{basename}_S11.npz")
dataS21 = np.load(f"../data/{basename}_S21.npz")
freqs = dataS11["freqs"] # Hz
S11 = dataS11["data_I"] + 1j*dataS11["data_Q"]
S21 = dataS21["data_I"] + 1j*dataS21["data_Q"]

df = (freqs[-1] - freqs[0]) / len(freqs)
peaks, _ = find_peaks(np.abs(S21), height=np.abs(S21)[0]*100, distance=500e6/df)
print(freqs[peaks])

for id in peaks:
    idx_min = id - math.floor(125e6/df)
    idx_max = id + math.floor(125e6/df)
    res = cr.fit_resonance(freqs[idx_min:idx_max], S11[idx_min:idx_max], S21[idx_min:idx_max], 
        plot_S11_real = False,
        plot_S11_imag = False,
        plot_S11_abs = False,
        plot_S11_phase = False,
        plot_S21_real = False,
        plot_S21_imag = False,
        plot_S21_abs = True,
        plot_S21_phase = True,
    )