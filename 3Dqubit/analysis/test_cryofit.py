import numpy as np
import cryofit as cr

basename = f"empty_cavity_3PC/65mK_1.0kHz_-5dBm"
dataS11 = np.loadtxt(f"../data/{basename}_S11.csv", delimiter=",")
dataS21 = np.loadtxt(f"../data/{basename}_S21.csv", delimiter=",")
freqs = dataS11[:,0] # Hz
S11 = dataS11[:,1] + 1j*dataS11[:,2]
S21 = dataS21[:,1] + 1j*dataS21[:,2]

print(dir(cr))

res = cr.fit_resonance(freqs, S11, S21, 
    plot_S11_real = False,
    plot_S11_imag = False,
    plot_S21_real = True,
    plot_S21_imag = False,
)