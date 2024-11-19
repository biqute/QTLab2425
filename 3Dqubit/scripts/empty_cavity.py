import matplotlib.pyplot as plt
import numpy as np

import sys; sys.path.append("../classes")
from VNA import VNA

# N9916A
myVNA = VNA("192.168.40.10")
myVNA.min_freq = 6e9
myVNA.max_freq = 8e9
myVNA.point_count = 2001
myVNA.timeout = 20e3
myVNA.bandwidth = 10e3
myVNA.avg_count = 50

datax = myVNA.read_frequency_data()
datay = myVNA.read_data("S21")

data_magn = 10*np.log10(np.square(datay["real"]) + np.square(datay["imag"]))

fig = plt.figure()
plt.plot(datax, data_magn)
plt.title("Cavity S21")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude [dB]")
plt.grid()

fig.set_size_inches(6, 5)
plt.savefig('..\\plots\\empty_cavity_6to8_S21.png')
plt.show()
