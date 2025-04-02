import numpy as np

import sys; sys.path.append("../classes")
from AWG import AWG

def gaussian(mu, sigma):
    denom = 2.0*sigma**2
    return lambda x: np.exp(-(x - mu)**2/denom)

mu = 1e-6
sigma = 1e-7

# SDG6052X
myAWG = AWG("192.168.3.15") # Check IP: Utility > Interface > LAN Setup > IP Address
myAWG.timeout = 10e3
myAWG.upload_waveform(
    name = "test", 
    func = gaussian(mu, sigma), 
    duration = 2*mu, 
    samples_per_second = 200e6
)
# myAWG.modulation = True
# myAWG.modulation_shape = ""
# myAWG.output = True

# import matplotlib.pyplot as plt
# datax = np.linspace(0, 2e-6)
# plt.scatter(datax, gaussian(1e-6, 1e-7)(datax))
# plt.xlim([0, 2e-6])
# plt.show()