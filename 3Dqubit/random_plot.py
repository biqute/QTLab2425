import matplotlib.pyplot as plt
import numpy as np


N = 10
datax = np.random.rand(N)
datay = np.random.rand(N)

print(datax)
print(datay)

plt.scatter(datax, datay)
plt.title("Random Plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()