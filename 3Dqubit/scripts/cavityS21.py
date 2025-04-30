import matplotlib.pyplot as plt
import numpy as np
import datetime

import sys; sys.path.append("../classes")
from VNA import VNA

# N9916A
myVNA = VNA("192.168.40.10")
#myVNA.min_freq = 7.46143442e9 - 50e6
#myVNA.max_freq = 7.46143442e9 + 50e6
myVNA.min_freq = 7.4297463e9 - 500e6
myVNA.max_freq = 7.4297463e9 + 500e6
myVNA.point_count = 10000
myVNA.timeout = 360e3
myVNA.avg_count = 10
myVNA.bandwidth = 1000 # Hz
myVNA.power = -40 # dBm

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
    # basename = f"../data/empty_cavity_S11_S12_S21_S22/40mK_{myVNA.bandwidth / 1e3}kHz_{myVNA.power}dBm_{Sij}"
    basename = f"../data/empty_cavity_3PC/300K_{myVNA.bandwidth / 1e3}kHz_{myVNA.power}dBm_S21_wide"
    
    np.savetxt(
        f"{basename}.csv", 
        np.transpose([dataf, data[Sij]["real"], data[Sij]["imag"]]), 
        delimiter=","
    )

    # Metadata
    metadata = (
        f"info, empty cavity, three port configuration for S21 = H_i + H_o\n" +
        #f"info, empty cavity, three port configuration for S21 = H_i + H_o\n" +
        f"time, {str(datetime.datetime.now())}\n" +
        f"temperature, 300K\n" +
        f"min_freq, {myVNA.min_freq}Hz\n" +
        f"max_freq, {myVNA.max_freq}Hz\n" +
        f"point_count, {myVNA.point_count}\n" +
        f"timeout, {myVNA.timeout}ms\n" +
        f"avg_count, {myVNA.avg_count}\n" +
        f"bandwidth, {myVNA.bandwidth}Hz\n" +
        f"power, {myVNA.power}dBm"
    )
    file = open(f"{basename}_meta.csv", 'w')
    file.write(metadata)
    file.close()