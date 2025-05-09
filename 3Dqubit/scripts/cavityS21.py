import matplotlib.pyplot as plt
import numpy as np
import datetime

import sys; sys.path.append("../classes")
from VNA import VNA
import sys; sys.path.append("../utils")
from dictionary_to_csv import dictionary_to_csv
from read_temperature import read_temperature

T = 64.6e-3 # read_temperature() # K

# N9916A
myVNA = VNA("192.168.40.10")
# myVNA.min_freq = 7.4297463e9 - 500e6 # hot
# myVNA.max_freq = 7.4297463e9 + 500e6 # hot
myVNA.min_freq = 7.4612500e9 - 500e6 # cold
myVNA.max_freq = 7.4612500e9 + 500e6 # cold
myVNA.point_count = 10000
myVNA.timeout = 360e3
myVNA.avg_count = 10
myVNA.bandwidth = 1000 # Hz
myVNA.power = -10 # dBm

data = { 
    # "S11": myVNA.read_data("S11"),
    # "S12": myVNA.read_data("S12"),
    "S21": myVNA.read_data("S21"),
    # "S22": myVNA.read_data("S22")
}
dataf = myVNA.read_frequency_data()

# Actual Data
# now = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-").split(".")[0]
# basename = f"../data/empty_cavity/unnamed_{now}"

for Sij in ["S21"]: #["S11", "S12", "S21", "S22"]:
    # basename = f"../data/empty_cavity_3PC/{int(T * 1e3)}mK_{myVNA.bandwidth / 1e3}kHz_{myVNA.power}dBm_S11"
    basename = f"../data/empty_cavity_3PC/{int(round(T * 1e3))}mK_{myVNA.bandwidth / 1e3}kHz_{myVNA.power}dBm_S21_wide"
    
    np.savetxt(
        f"{basename}.csv", 
        np.transpose([dataf, data[Sij]["real"], data[Sij]["imag"]]), 
        delimiter=","
    )

    # Metadata
    metadata = {
        #"info": "empty cavity, three port configuration for S11 = S_1 + H_o", 
        "info": "empty cavity, three port configuration for S21 = H_i + H_o", 
        "time": f"{str(datetime.datetime.now())}", 
        "temperature": f"{T}K", 
        "min_freq": f"{myVNA.min_freq}Hz", 
        "max_freq": f"{myVNA.max_freq}Hz", 
        "point_count": f"{myVNA.point_count}", 
        "timeout": f"{myVNA.timeout}ms", 
        "avg_count": f"{myVNA.avg_count}", 
        "bandwidth": f"{myVNA.bandwidth}Hz", 
        "power": f"{myVNA.power}dBm"
    }
    file = open(f"{basename}_meta.csv", 'w')
    file.write(dictionary_to_csv(metadata))
    file.close()

