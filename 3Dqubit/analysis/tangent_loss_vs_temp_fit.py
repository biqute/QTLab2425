import numpy as np
import cmath
import sys; sys.path.append("../classes")
from Fitter import Fitter
import sys; sys.path.append("../utils")
from peak_width import peak_width
import math
import matplotlib.pyplot as plt

# |S_21(f)|
# https://arxiv.org/pdf/1410.3365 eq. (1)
def model_modulus_notch(f, a, b, c, d, A, phi, Q_l, Q_c, f_r):
    df = f - f_r
    y = a + b*df + c*df**2 + d*df**3 + A*np.abs(
        1 - Q_l * cmath.exp(1j*phi) / np.abs(Q_c) / (
            1 + 2j * Q_l * df / f_r
        )
    )
    return y


fitted_freqs = np.array([])
fitted_Qcs = np.array([])
fitted_Qls = np.array([])

N = int((116 - 40) / 2)
n = 40
T = 410e-3 # K
Ts = np.array([])
for _ in range(0, N):
    Ts = np.append(Ts, T)
    data = np.loadtxt(f"..\\data\\gap_run12_iq\\Q_res{n}.txt", delimiter=",")

    fitter = Fitter()
    fitter.datax = data[:, 0] # f [Hz]
    fitter.datay = np.sqrt(data[:, 1]**2 + data[:, 2]**2) # |S21|
    fitter.model = model_modulus_notch
    
    # ESTIMATION OF FIT PARAMETERS

    Q_c = 50e3 # coupling quality factor
    f_r = fitter.datax[np.argmin(fitter.datay)]
    width = peak_width(fitter.datax, -fitter.datay)
    Q_i = f_r / width # internal quality factor
    Q_l = 1/(1/Q_c + 1/Q_i) # loaded quality factor
    A = ( np.max(fitter.datay) - np.min(fitter.datay) ) * Q_c / Q_l
    a = fitter.datay[0] - fitter.model(fitter.datax[0], 0.0, b = 0, c = 0, d = 0, A = A, phi = 0, Q_l = Q_l, Q_c = Q_c, f_r = f_r)

    fitter.params = {
        "A":   (0.0, A, None),
        "a":   (None, a, None),
        "b":   (None, 0.0, None),
        "c":   (None, 0.0, None),
        "d":   (None, 0.0, None),
        "phi": (-math.pi, 0, math.pi), 
        "Q_l": (0, Q_l, None), 
        "Q_c": (0, Q_c, None), 
        "f_r": (0, f_r, None)
    }

    # CROP DATA

    deltaf = ( fitter.datax[-1] - fitter.datax[0] ) / len(fitter.datax)
    min_f = f_r - 2.5*width
    max_f = f_r + 2.5*width
    min_f_idx = int((min_f - fitter.datax[0]) / deltaf)
    max_f_idx = int((max_f - fitter.datax[0]) / deltaf)
    min_f_idx = max(0, min_f_idx)
    max_f_idx = min(len(fitter.datax) - 1, max_f_idx)

    fitter.datax = fitter.datax[min_f_idx:max_f_idx]
    fitter.datay = fitter.datay[min_f_idx:max_f_idx]
    fitter.sigmay = np.maximum(1e-5 + fitter.datax*0, np.abs(fitter.datay*0.01)) # TODO

    # ACTUALLY FIT

    res = fitter.fit()
    fitted_Qcs = np.append(fitted_Qcs, res["params"]["Q_c"]["value"])
    fitted_Qls = np.append(fitted_Qls, res["params"]["Q_l"]["value"])
    fitted_freqs = np.append(fitted_freqs, res["params"]["f_r"]["value"])

    n += 2
    T -= 10e-3

fitted_Qis = (fitted_Qls * fitted_Qcs) / (fitted_Qcs - fitted_Qls)

#plt.plot(Ts, fitted_freqs, label="$f_r$")
plt.plot(Ts, 1 / fitted_Qis, label="$Q_i^{-1}$")
plt.title("$Q_i^{-1}$ vs. T")
plt.xlabel("T [mK]")
plt.ylabel("1 / Q")
plt.legend()
plt.grid()
plt.show()