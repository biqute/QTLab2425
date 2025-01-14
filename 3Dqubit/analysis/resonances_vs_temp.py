import numpy as np
import matplotlib.pyplot as plt

N = int((116 - 40) / 2)
n = 40
T = 410e-3 # K
dT = 10e-3
Tmax = T
Tmin = T - N*dT
for _ in range(0, N + 1):
    data = np.loadtxt(f"..\\data\\gap_run12_iq\\Q_res{n}.txt", delimiter=",")
    datax = data[:, 0] # f [Hz]
    datay = np.sqrt(data[:, 1]**2 + data[:, 2]**2) # |S21|
    
    x = (T - Tmin) / (Tmax - Tmin)
    x = x**4
    color = x * np.array([1,0,0]) + (1 - x) * np.array([0,0,1])
    
    plt.plot(datax, datay, label=f"T = {T*1000}mK", color=(
        max(0, min(1, float(color[0]))), 
        max(0, min(1, float(color[1]))), 
        max(0, min(1, float(color[2]))),
        0.5
    ))

    n += 2
    T -= dT

plt.title(f"Resonances from Tmin = {Tmin*1000:.3}mK (blue) to Tmax = {Tmax*1000}mK (red)")
plt.grid()
plt.xlim([3.1765e9, 3.1785e9])
#plt.legend()
plt.savefig('..\\plots\\resonances_vs_temp.png')
plt.show()
