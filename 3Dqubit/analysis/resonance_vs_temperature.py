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


Ts = np.array([20, 150, 300, 500]) # mK
dBms = np.array([-20, -20, -20, -20]) # dBm
fitted_freqs = np.array([])
fitted_Qcs = np.array([])
fitted_Qls = np.array([])

for T, dBm in zip(Ts, dBms):
    data = np.loadtxt(f"..\\data\\resonances\\{T}mK_{dBm}dBm.txt")

    fitter = Fitter()
    fitter.datax = data[:, 0]
    fitter.datay = np.power(10 + fitter.datax*0, data[:, 1]/20)
    fitter.sigmay = np.maximum(1e-5 + fitter.datax*0, np.abs(fitter.datay*0.01))
    fitter.model = model_modulus_notch

    # ESTIMATION OF FIT PARAMETERS

    Q_c = 2e3 # coupling quality factor
    f_r = fitter.datax[np.argmin(fitter.datay)]
    Q_i = f_r / peak_width(fitter.datax, -fitter.datay) # internal quality factor
    Q_l = 1/(1/Q_c + 1/Q_i) # loaded quality factor
    A = 1.0
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

    res = fitter.fit()
    fitted_Qcs = np.append(fitted_Qcs, res["params"]["Q_c"]["value"])
    fitted_Qls = np.append(fitted_Qls, res["params"]["Q_l"]["value"])
    fitted_freqs = np.append(fitted_freqs, res["params"]["f_r"]["value"])

fitted_Qis = (fitted_Qls * fitted_Qcs) / (fitted_Qcs - fitted_Qls)

#plt.plot(Ts, fitted_freqs, label="$f_r$")
plt.plot(Ts, 1 / fitted_Qis, label="$Q_i^{-1}$")
plt.title("$Q_i^{-1}$ vs. T")
plt.xlabel("T [mK]")
plt.ylabel("1 / Q")
plt.legend()
plt.grid()
plt.show()