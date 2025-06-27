import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.special import iv, kv
from scipy.constants import hbar, k

def read_data(filename, delimiter = None):
    if delimiter == ' ':
        data = np.loadtxt(filename)
    else:
        data = np.loadtxt(filename, delimiter=delimiter)
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

    # Plot del fit circolare
    plot_circle_fit(z_nodelay, xc, yc, r)

    z_rot, angle = rotate_and_center(z_nodelay, xc, yc)
    fr, Qt, theta0 = fit_phase_minuit(f, z_rot)
    Qc, Qi = calc_quality_factors(xc, yc, r, Qt)

    # Stampa dei risultati
    print("\n=== Risultati del fit ===")
    print(f"fr  = {fr:.6f} Hz")
    print(f"Qt  = {Qt:.2f}")
    print(f"Qc  = {Qc:.2f}")
    print(f"Qi  = {Qi:.2f}")

    # Controllo sulla relazione tra Qt e Qc
    if Qt >= Qc:
        print("⚠️  Warning: Qt >= Qc. Possibile errore nel fit o accoppiamento sovracritico.")
    else:
        print("✅ Verifica: Qt < Qc (ok)")

    return Qc, Qi, Qt, fr


def plot_circle_fit(z, xc, yc, r):
    fig, ax = plt.subplots()
    ax.plot(z.real, z.imag, 'o', label='Dati misurati', markersize=3)
    
    theta = np.linspace(0, 2*np.pi, 400)
    circle_x = xc + r * np.cos(theta)
    circle_y = yc + r * np.sin(theta)
    ax.plot(circle_x, circle_y, 'r-', label='Fit circolare')

    ax.plot(xc, yc, 'kx', label='Centro del cerchio')
    ax.set_aspect('equal')
    ax.set_xlabel("Re[Z]")
    ax.set_ylabel("Im[Z]")
    plt.title("Fit circolare del dato complesso (IQ)")
    ax.legend()
    plt.grid(True)
    plt.show()
