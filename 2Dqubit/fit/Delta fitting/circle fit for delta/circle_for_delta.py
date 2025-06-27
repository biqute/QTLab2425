import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.special import iv, kv
from scipy.constants import hbar, k, h 

def read_data(filename):
    data = np.loadtxt(filename, delimiter=",")
    f = data[:, 0]
    I = data[:, 1]
    Q = data[:, 2]
    z = I + 1j * Q
    return f, z

def remove_cable_delay(f, z):
    phase = np.unwrap(np.angle(z))
    p = np.polyfit(f, phase, 1)
    delay_phase = np.exp(-1j * np.polyval(p, f))
    return z * delay_phase

def fit_circle(x, y):
    x_m = np.mean(x)
    y_m = np.mean(y)
    u = x - x_m
    v = y - y_m
    Suu = np.sum(u**2)
    Suv = np.sum(u*v)
    Svv = np.sum(v**2)
    Suuu = np.sum(u**3)
    Suvv = np.sum(u*v**2)
    Svvv = np.sum(v**3)
    Svuu = np.sum(v*u**2)

    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([(Suuu + Suvv)/2.0, (Svvv + Svuu)/2.0])
    uc, vc = np.linalg.solve(A, B)
    xc = x_m + uc
    yc = y_m + vc
    r = np.mean(np.sqrt((x - xc)**2 + (y - yc)**2))
    return xc, yc, r

def rotate_and_center(z, xc, yc):
    zc = xc + 1j * yc
    angle = np.angle(zc)
    return (z - zc) * np.exp(-1j * angle), angle

def phase_model(f, fr, Qt, theta0):
    return theta0 + 2 * np.arctan(2 * Qt * (1 - f / fr))

def fit_phase_minuit(f, z_rot):
    phase = np.unwrap(np.angle(z_rot))
    fr0 = f[np.argmin(np.abs(np.abs(z_rot) - np.min(np.abs(z_rot))))]
    cost = LeastSquares(f, phase, np.ones_like(phase), phase_model)
    minuit = Minuit(cost, fr=fr0, Qt=10000, theta0=0)
    minuit.migrad()
    return minuit.values['fr'], minuit.values['Qt'], minuit.values['theta0']

def calc_quality_factors(xc, yc, r, Qt):
    Qc = ((np.sqrt(xc**2 + yc**2) + r) / (2 * r)) * Qt
    Qi = 1 / (1/Qt - 1/Qc)
    return Qc, Qi

def fit_full_resonator(filename):
    f, z = read_data(filename)
    z_nodelay = remove_cable_delay(f, z)
    xc, yc, r = fit_circle(z_nodelay.real, z_nodelay.imag)
    z_rot, angle = rotate_and_center(z_nodelay, xc, yc)
    fr, Qt, theta0 = fit_phase_minuit(f, z_rot)
    Qc, Qi = calc_quality_factors(xc, yc, r, Qt)
    return Qc, Qi, Qt, fr

# Funzione teorica per il fit della Delta

def model(T, fr, Q0, Delta0_meV, alpha=0.8):
    meV_to_J = 1.60218e-22  # 1 meV in Joules
    Delta0 = Delta0_meV * meV_to_J
    T = np.asarray(T, dtype=float) * 1e-3  # conversione mK -> K
    
    xi = (hbar * fr) / (2 * k * T)  
    bessel_term = np.sinh(xi) * kv(0, xi)  
    iv_term = np.exp(-xi) * iv(0, -xi)

    num = np.exp(-Delta0 / (k * T)) * bessel_term
    den = 1 - 2 * np.exp(-Delta0 / (k * T)) * iv_term
    
    return (1/Q0 + (2 * alpha / np.pi) * (num / den))

def fit(Qi, T, freq, alpha_fixed=0.8):
    T = np.asarray(T, dtype=float)
    Qi = np.asarray(Qi, dtype=float)
    Qi_err = 0.01 * Qi  # errore relativo 1%
    inv_Qi_err = Qi_err / Qi**2

    least_squares = LeastSquares(T, 1/Qi, inv_Qi_err, model)
    minuit = Minuit(least_squares,
                    fr=freq,
                    Q0=max(Qi),
                    Delta0_meV=0.3,
                    alpha=alpha_fixed)

    minuit.limits["Q0"] = (1e3, None)
    minuit.limits["Delta0_meV"] = (0.14, 0.5)  # vincolo realistico per Al
    minuit.fixed["fr"] = True
    minuit.fixed["alpha"] = True

    minuit.migrad()
    minuit.hesse()

    print("Successo del fit:", minuit.valid)
    print("Chi-quadro ridotto:", minuit.fval / minuit.ndof)
    for par, val, err in zip(minuit.parameters, minuit.values, minuit.errors):
        print(f"{par} = {val:.4f} ± {err:.4f}")

    Q0_fit = minuit.values["Q0"]
    Delta0_fit = minuit.values["Delta0_meV"]

    fig, ax = plt.subplots()
    ax.plot(T, 1/Qi, "bo", label="Dati sperimentali", markersize=3)
    ax.plot(T, model(T, freq, Q0_fit, Delta0_fit), "r-", label="Fit")
    ax.set_xlabel("Temperatura (mK)")
    ax.set_ylabel("1/Qi")
    ax.legend()
    plt.title("Fit della qualità interna")
    plt.show()

    return Q0_fit, Delta0_fit
