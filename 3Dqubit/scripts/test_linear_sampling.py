import matplotlib.pyplot as plt
import numpy as np
import sys; sys.path.append("../utils")
from linear_sampling import linear_sampling

datax = np.array([0, 1, 4, 2, 5, 6])
datay = np.array([3, 6, 2, 8, 3, 4])
samplesx = np.linspace(-2, 10, 100)

fig = plt.figure()
plt.scatter(datax, datay)
plt.plot(samplesx, linear_sampling(samplesx, datax, datay))
plt.grid()
plt.show()