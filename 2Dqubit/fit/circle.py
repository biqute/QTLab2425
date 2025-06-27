import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.special import iv, kv
from scipy.constants import hbar, k


def read_data(filename, delimiter=' '):
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


def apply_cut(f, z, cut_factor):
    mag = np.abs(z)
    threshold = np.min(mag) + cut_factor * (np.max(mag) - np.min(mag))
    mask = mag <= threshold
    return f[mask], z[mask]


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


def fit_phase_minuit(f, z_rot, initial_params):
    phase = np.unwrap(np.angle(z_rot))
    cost = LeastSquares(f, phase, np.ones_like(phase), phase_model)
    minuit = Minuit(cost,
                    fr=initial_params['fr'],
                    Qt=initial_params['Qt'],
                    theta0=initial_params['phi'])
    minuit.limits['Qt'] = (0, initial_params['Qc'])
    minuit.limits['fr'] = (initial_params['fr'] * 0.95, initial_params['fr'] * 1.05)
    minuit.migrad()
    phase_fit = phase_model(f, minuit.values['fr'], minuit.values['Qt'], minuit.values['theta0'])
    return minuit.values['fr'], minuit.values['Qt'], minuit.values['theta0'], phase, phase_fit


def calc_quality_factors(xc, yc, r, Qt):
    Qc = ((np.sqrt(xc**2 + yc**2) + r) / (2 * r)) * Qt
    Qi = 1 / (1/Qt - 1/Qc)
    return Qc, Qi


def plot_results(f, z, z_rot, xc, yc, r, phase, phase_fit, fr, Qt, Qc, Qi):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].plot(z.real, z.imag, '.', label='Dati tagliati')
    circle = plt.Circle((xc, yc), r, color='r', fill=False, label='Fit circolare')
    axs[0, 0].add_artist(circle)
    axs[0, 0].set_title('Cerchio nel piano IQ')
    axs[0, 0].axis('equal')
    axs[0, 0].legend()

    axs[0, 1].plot(f, np.abs(z), '.', label='Modulo z')
    axs[0, 1].set_title('Modulo della risposta')
    axs[0, 1].set_xlabel('Frequenza [Hz]')

    axs[1, 0].plot(f, phase, '.', label='Fase sperimentale')
    axs[1, 0].plot(f, phase_fit, '-', label='Fit fase')
    axs[1, 0].set_title('Fit della fase')
    axs[1, 0].legend()
    axs[1, 0].set_xlabel('Frequenza [Hz]')
    axs[1, 0].set_ylabel('Fase [rad]')

    axs[1, 1].plot(f, phase - phase_fit, '.', label='Residui fase')
    axs[1, 1].set_title('Residui del fit di fase')
    axs[1, 1].set_xlabel('Frequenza [Hz]')
    axs[1, 1].set_ylabel('Residuo [rad]')

    for ax in axs.flat:
        ax.grid(True)

    plt.tight_layout()
    plt.suptitle(f"Fit completo\nfr={fr:.2e} Hz, Qt={Qt:.2f}, Qc={Qc:.2f}, Qi={Qi:.2f}", y=1.02)
    plt.show()


def fit_full_resonator(filename, initial_params, cut_factor=0.2, delimiter=' '):
    f, z = read_data(filename, delimiter)
    z_nodelay = remove_cable_delay(f, z)
    f_cut, z_cut = apply_cut(f, z_nodelay, cut_factor)
    xc, yc, r = fit_circle(z_cut.real, z_cut.imag)
    z_rot, angle = rotate_and_center(z_cut, xc, yc)
    fr, Qt, theta0, phase, phase_fit = fit_phase_minuit(f_cut, z_rot, initial_params)
    Qc, Qi = calc_quality_factors(xc, yc, r, Qt)
    plot_results(f_cut, z_cut, z_rot, xc, yc, r, phase, phase_fit, fr, Qt, Qc, Qi)
    return Qc, Qi, Qt, fr
