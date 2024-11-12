import matplotlib.pyplot as plt
import numpy as np
from PSA import PSA

SETTINGS = {
    "points": 400,
    "min_freq": 4e9, # Hz
    "max_freq": 6e9, # Hz
    "timeout": 10e3, # ms
}

myPSA = PSA("tcpip0::192.168.40.10::INSTR")

myPSA.set_timeout(SETTINGS["timeout"])
myPSA.set_min_freq(SETTINGS["min_freq"])
myPSA.set_max_freq(SETTINGS["max_freq"])
myPSA.set_point_count(SETTINGS["points"])

datay = myPSA.read_data()

"""
# N9916A
rm_PSA = pyvisa.ResourceManager()
PSA = rm_PSA.open_resource("tcpip0::192.168.40.10::INSTR")

PSA.timeout = SETTINGS["timeout"]
PSA.write("*CLS")
print(PSA.query("*IDN?"))
PSA.query("INST:SEL 'SA';*OPC?") # SA (Spectrum Analyzer), CAT, NA

#print("Preset complete: " + PSA.query("SYST:PRES;*OPC?"))

PSA.write("SENS:SWE:POIN " + str(SETTINGS["points"]))
PSA.write("SENS:FREQ:START " + str(SETTINGS["min_freq"]))
PSA.write("SENS:FREQ:STOP " + str(SETTINGS["max_freq"]))

PSA.query("FORM COMP,32; *OPC?")
# print(PSA.query("TRACE1:TYPE?"))

# PSA.query("INIT:CONT OFF; *OPC?")
# print("Trigger complete: " + PSA.query("INIT:IMM; *OPC?")) # Trigger
datay = list(map(float, PSA.query("TRACE:DATA?").split(",")))
PSA.query("*OPC?")
"""

datax = np.linspace(SETTINGS["min_freq"],SETTINGS["max_freq"],SETTINGS["points"])
plt.plot(datax, datay)
plt.title("Random Plot")
plt.xlabel("Frequency (Hz)")
plt.ylabel("dB")
plt.grid()
plt.show()



# SMA100B
#rm_1 = pyvisa.ResourceManager()
#resource_1 = rm_1.open_resource('tcpip0::192.168.40.15::inst0::INSTR')
#print(resource_1.query("*IDN?"))

