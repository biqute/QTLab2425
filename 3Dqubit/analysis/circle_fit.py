import numpy as np
import cmath
import matplotlib.pyplot as plt
import sys; sys.path.append("../classes")
import sys; sys.path.append("../utils")
from peak_width import peak_width
from Fitter import Fitter
import math

# data = np.loadtxt("..\\data\\gap_run12_iq\\Q_res40.txt", delimiter=",")
data = np.loadtxt("..\\data\\cableS21.csv", delimiter=",")

plt.scatter(data[:, 1], data[:, 2])
plt.title("Circle")
plt.xlabel("$\\text{Re}(S_{12})$")
plt.ylabel("$\\text{Im}(S_{12})$")
plt.show()