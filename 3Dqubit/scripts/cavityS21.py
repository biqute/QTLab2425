import matplotlib.pyplot as plt
import numpy as np

import sys; sys.path.append("../classes")
from VNA import VNA

# N9916A
myVNA = VNA("192.168.40.10")
myVNA.min_freq = 7.2e9
myVNA.max_freq = 7.6e9
myVNA.point_count = 10000
myVNA.timeout = 180e3
myVNA.avg_count = 15

dataS21 = myVNA.read_data("S21")
dataf = myVNA.read_frequency_data()

np.savetxt("../data/dense_cavityS21.csv", np.transpose([dataf, dataS21["real"], dataS21["imag"]]), delimiter=",")