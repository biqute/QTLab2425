import matplotlib.pyplot as plt
import numpy as np
import pyvisa
import math
from VNA import VNA

# SMA100B
myVNA = VNA("192.168.40.10")
myVNA.min_freq = 3.5e9
myVNA.max_freq = 7e9
myVNA.point_count = 500
myVNA.timeout = 10e3

"""
# SMA100B
rm = pyvisa.ResourceManager()
VNA = rm.open_resource("tcpip0::192.168.40.10::INSTR")
VNA.timeout = SETTINGS["timeout"]
print(VNA.query("*IDN?"))
VNA.write("*CLS")
VNA.write("*RST")
VNA.query("INST:SEL 'NA'; *OPC?")

VNA.write("SENS:SWE:POIN " + str(SETTINGS["points"]))
VNA.write("SENS:FREQ:START " + str(SETTINGS["min_freq"]))
VNA.write("SENS:FREQ:STOP " + str(SETTINGS["max_freq"]))

VNA.query("INIT:CONT 0;*OPC?")

data = {}

VNA.query("CALC:PAR:DEF S11; *OPC?")
VNA.query("INIT:IMMediate;*OPC?")
data["S11"] = list(map(float, VNA.query("CALC:DATA:SDATA?").split(",")))
VNA.query("CALC:PAR:DEF S12; *OPC?")
VNA.query("INIT:IMMediate;*OPC?")
data["S12"] = list(map(float, VNA.query("CALC:DATA:SDATA?").split(",")))
VNA.query("CALC:PAR:DEF S21; *OPC?")
VNA.query("INIT:IMMediate;*OPC?")
data["S21"] = list(map(float, VNA.query("CALC:DATA:SDATA?").split(",")))
VNA.query("CALC:PAR:DEF S22; *OPC?")
VNA.query("INIT:IMMediate;*OPC?")
data["S22"] = list(map(float, VNA.query("CALC:DATA:SDATA?").split(",")))

VNA.query("INIT:CONT 1;*OPC?")


datax = np.linspace(SETTINGS["min_freq"],SETTINGS["max_freq"],SETTINGS["points"])

Snames = [["S11", "S12"], ["S21", "S22"]]

fig, axes = plt.subplots(2, 2)
for i in [0,1]:
    for j in [0,1]:
        Sij = Snames[i][j]
        data_real = data[Sij][0::2] # all even entries 0,2,4,6,8
        data_imag = data[Sij][1::2] # all odd entries 1,3,5,7,9
        data_magn = [10*math.log10(data_real[i]**2 + data_imag[i]**2) for i in range(0, len(data_real))] 
        axes[i,j].plot(datax, data_magn)
        axes[i,j].set_title(Sij)
        axes[i,j].grid()
fig.suptitle("VNA")

for ax in axes.flat:
    ax.set(xlabel="Frequency [Hz]", ylabel="Amplitude [dB]")
for ax in axes.flat:
    ax.label_outer()

plt.show()
"""
