import os
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
import cmath


# |S_21(f)|
# https://arxiv.org/pdf/1410.3365 eq. (1)
def model_modulus_notch(f, a, phi, Q_l, Q_c, f_r):
    return a*abs(
        1 - Q_l * cmath.exp(1j*phi) / abs(Q_c) / (
            1 + 2j * Q_l * (f/f_r - 1)
        )
    )

data = np.loadtxt("..\\data\\resonances\\20mK_-5dBm.txt")

datax = data[:, 0]
datay = data[:, 1]


least_squares = LeastSquares(datax, datay, 0.01*datay, model_modulus_notch)

start_values = {
    "a": -9.0, 
    "phi": 3.14/3, 
    "Q_l": 1e3, 
    "Q_c": 3e3, 
    "f_r": 5.4e9
}
m = Minuit(least_squares, **start_values)  # starting values for α and β
m.migrad()
m.hesse()

text = ""
for p, v, e in zip(m.parameters, m.values, m.errors):
    text += f"{p} = {v:.3f} ± {e:.3f}\n"
text = text.strip()
print(text)

freqs = np.linspace(np.min(datax), np.max(datax), 2000)
modely = [model_modulus_notch(f, *m.values) for f in freqs]
starty = [model_modulus_notch(f, **start_values) for f in freqs]

plt.plot(datax, datay)
plt.plot(freqs, modely)
plt.plot(freqs, starty)
plt.title("Resonance plot")
plt.xlabel("Frequency (Hz)")
plt.ylabel("dB")
plt.text(0.5, 0.5, text, fontsize=14)
plt.grid()
plt.show()

