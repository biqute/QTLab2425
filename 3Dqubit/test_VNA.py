import matplotlib.pyplot as plt
import numpy as np
import math
from VNA import VNA

# N9916A
myVNA = VNA("192.168.40.10")
myVNA.min_freq = 3.5e9
myVNA.max_freq = 7e9
myVNA.point_count = 500
myVNA.timeout = 20e3
myVNA.bandwidth = 10e3
myVNA.avg_count = 5

data = {
    "S11": myVNA.read_data("S11"),
    "S12": myVNA.read_data("S12"),
    "S21": myVNA.read_data("S21"),
    "S22": myVNA.read_data("S22"),
}

datax = myVNA.read_frequency_data()

Snames = [["S11", "S12"], ["S21", "S22"]]

fig, axes = plt.subplots(2, 2)
for i in [0,1]:
    for j in [0,1]:
        Sij = Snames[i][j]
        data_magn = np.array([10*math.log10(data[Sij]["real"][i]**2 + data[Sij]["imag"][i]**2) for i in range(0, myVNA.point_count)])
        axes[i,j].plot(datax, data_magn)
        axes[i,j].set_title(Sij)
        axes[i,j].grid()
fig.suptitle("VNA")

for ax in axes.flat:
    ax.set(xlabel="Frequency [Hz]", ylabel="Amplitude [dB]")

plt.show()
