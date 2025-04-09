import numpy as np

import sys; sys.path.append("../classes")
from AWG import AWG

def gaussian(mu, sigma):
    denom = 2.0*sigma**2
    return lambda x: np.exp(-(x - mu)**2/denom)

def modulated_harmonic(A, f, phi, mu, sigma):
    denom = 2.0*sigma**2
    return lambda x: A * np.sin(2*np.pi*f*x + phi) * np.exp(-(x - mu)**2/denom)

T = 1e-6
sigma = 1e-7

# SDG6052X
myAWG = AWG("192.168.3.15") # Check IP: Utility > Interface > LAN Setup > IP Address
myAWG.timeout = 10e3

# myAWG.upload_waveform(
#     name = "TEST", 
#     # func = gaussian(0, sigma), 
#     func = modulated_harmonic(1, 15/T, 0, 0, sigma), 
#     interval = (-T/2, T/2), 
# )

myAWG.freq = 2e3 # 2kHz
myAWG.amplitude = 4 # Vpp
myAWG.waveform = "Local/TEST.bin"
myAWG.output = True

# import matplotlib.pyplot as plt
# datax = np.linspace(0, 2e-6)
# plt.scatter(datax, gaussian(1e-6, 1e-7)(datax))
# plt.xlim([0, 2e-6])
# plt.show()