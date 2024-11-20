import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks 

import sys; sys.path.append("../classes")
from VNA import VNA

# N9916A
myVNA = VNA("192.168.40.10")
myVNA.min_freq = 10e9
myVNA.max_freq = 12e9
myVNA.point_count = 2001
myVNA.timeout = 20e3
myVNA.bandwidth = 10e3
myVNA.avg_count = 20

datax = myVNA.read_frequency_data()
datay = myVNA.read_data("S21")

data_magn = 10*np.log10(np.square(datay["real"]) + np.square(datay["imag"]))

peaks, _ = find_peaks(data_magn, height = -75, width = 10)

fig = plt.figure()
plt.plot(datax, data_magn)
plt.plot(datax[peaks], data_magn[peaks], "x")
plt.title("Cavity S21")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude [dB]")
plt.grid()

fig.set_size_inches(6, 5)
# plt.savefig('..\\plots\\empty_cavity_6to8_S21.png')
plt.show()


