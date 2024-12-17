import numpy as np
import cmath
import sys; sys.path.append("../classes")
from Fitter import Fitter
import math

# |S_21(f)|
# https://arxiv.org/pdf/1410.3365 eq. (1)
def model_modulus_notch(f, a, b, A, phi, Q_l, Q_c, f_r):
    c = 0
    d = 0
    df = f - f_r
    y = a + b*df + c*df**2 + d*df**3 + A*np.abs(
        1 - Q_l * cmath.exp(1j*phi) / np.abs(Q_c) / (
            1 + 2j * Q_l * (f/f_r - 1)
        )
    )
    return y
    # return 20*np.log10(y)

data = np.loadtxt("..\\data\\resonances\\20mK_-5dBm.txt")

fitter = Fitter()
fitter.datax = data[:, 0]
fitter.datay = np.power(10 + fitter.datax*0, data[:, 1]/20) # data[:, 1]
fitter.sigmay = np.maximum(1e-5 + fitter.datax*0, np.abs(fitter.datay*0.01))
fitter.model = model_modulus_notch
fitter.show_initial_model = True
fitter.params = {
    "A":   (0.0, 0.27, None),
    "a":   (None, 0.15 - 0.075, None),
    "b":   (None, 0.0, None),
    # "c":   (None, 10**(-18/20), None),
    # "d":   (None, 10**(-27/20), None),
    "phi": (-math.pi, 0.3, math.pi), 
    "Q_l": (0, 820, None), 
    "Q_c": (0, 2e3, None), 
    "f_r": (0, fitter.datax[np.argmin(fitter.datay)], None)
}
fitter.param_units = {
    "A":   "1", 
    "a":   "1", 
    "b":   "s", 
    "c":   "s^2", 
    "d":   "s^3", 
    "phi": "rad", 
    "Q_l": "1", 
    "Q_c": "1", 
    "f_r": "Hz"
}
fitter.param_displayed_names = { "phi": "\\phi" }
fitter.unitx = "Hz"
fitter.unity = "1"

fitter.plot_fit()

"""
f = abs(
    par(6)*data(1,:) 
    + par(7)*data(1,:).^2 
    + par(8)*data(1,:).^3 
    + par(1)*(1 - exp(1i*par(4))*par(2)*(par(2)^-1 - (par(3))^-1) ./ (1 + 2i * par(2) * ( data(1,:) - par(5) ) ./ fmin )));
"""

'''
least_squares = LeastSquares(datax, datay, 0.01*datay, model_modulus_notch)

start_values = {
    "a": -9.0, 
    "phi": 3.14/3, 
    "Q_l": 1e3, 
    "Q_c": 3e3, 
    "f_r": 5.4e9
}


m = Minuit(least_squares, **start_values)
m.limits["f_r"] = (2e9, 3e9)
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

'''