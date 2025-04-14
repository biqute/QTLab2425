import matplotlib.pyplot as plt
import numpy as np


Snames = [["S11", "S12"], ["S21", "S22"]]

basename = f"../data/empty_cavity_S11_S21_S22/40mK_1.0kHz_-10dBm"
dataS11 = np.loadtxt(f"{basename}_S11.csv", delimiter=",")
dataS21 = np.loadtxt(f"{basename}_S21.csv", delimiter=",")
dataS22 = np.loadtxt(f"{basename}_S22.csv", delimiter=",")
data = {
    "S11": {"real": dataS11[:,1], "imag": dataS11[:,2]},
    "S21": {"real": dataS21[:,1], "imag": dataS21[:,2]},
    "S22": {"real": dataS22[:,1], "imag": dataS22[:,2]},
}
datax = dataS11[:,0] # Hz

fig, axes = plt.subplots(2, 2)
for i in [0,1]:
    for j in [0,1]:
        Sij = Snames[i][j]
        if Sij not in data: continue
        data_magn = 10*np.log10(np.square(data[Sij]["real"]) + np.square(data[Sij]["imag"]))
        axes[i,j].plot(datax, data_magn)
        axes[i,j].set_title(Sij)
        axes[i,j].grid()
fig.suptitle("VNA")

for ax in axes.flat:
    ax.set(xlabel="Frequency [Hz]", ylabel="Amplitude [dB]")

fig.set_size_inches(12, 10)
#plt.savefig('plots\\triple_plot.pdf')
plt.show()