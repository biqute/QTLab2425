import numpy as np
import cmath
import sys; sys.path.append("../classes")
from Fitter import Fitter


# |S_21(f)|
# https://arxiv.org/pdf/1410.3365 eq. (1)
def model_modulus_notch(f, a, b, c, d, phi, Q_l, Q_c, f_r):
    return (a + b*(f-f_r) + c*(f-f_r)**2 + d*(f-f_r)**3)*abs(
        1 - Q_l * cmath.exp(1j*phi) / abs(Q_c) / (
            1 + 2j * Q_l * (f/f_r - 1)
        )
    )

data = np.loadtxt("..\\data\\resonances\\20mK_-5dBm.txt")

fitter = Fitter()
fitter.datax = data[:, 0]
fitter.datay = np.power(10 + fitter.datax*0, data[:, 1]/20)
fitter.sigmay = np.abs(fitter.datay*0.01)
fitter.model = model_modulus_notch
fitter.params = {
    "a":   (None, -9.0, None),
    "b":   (None, 0, None),
    "c":   (None, 0, None),
    "d":   (None, 0, None),
    "phi": (None, 3.14/3, None), 
    "Q_l": (0, 1e3, None), 
    "Q_c": (0, 3e3, None), 
    "f_r": (0, 5.4e9, None)
}
fitter.param_units = {
    "a":   "u", 
    "b":   "u", 
    "c":   "u", 
    "d":   "u", 
    "phi": "u", 
    "Q_l": "u", 
    "Q_c": "u", 
    "f_r": "u"
}
fitter.param_displayed_names = { "phi": "\\phi" }
fitter.unitx = "ux"
fitter.unity = "uy"

fitter.plot_fit()

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