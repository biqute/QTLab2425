import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.special import iv, kv
from scipy.constants import hbar, k, h
import matplotlib.cm as cm

# -----------------------------------------------
# Lettura dei dati complessi (f, I, Q) dal file CSV
# -----------------------------------------------
def read_data(filename):
    data = np.loadtxt(filename, delimiter=",")
    f = data[:, 0]      # Frequenza
    I = data[:, 1]      # Componente In-Phase
    Q = data[:, 2]      # Componente in quadratura
    z = I + 1j * Q      # Costruzione del segnale complesso z = I + iQ
    return f, z


# -----------------------------------------------
# Rimozione della fase dovuta al ritardo dei cavi
# Rif. Appendice E.2.1
# -----------------------------------------------
def remove_cable_delay(f, z, window=0.2):
    n = len(f)
    n_win = int(n * window)

    # Estrai le frequenze e la fase solo agli estremi della banda
    f_fit = np.concatenate([f[:n_win], f[-n_win:]])
    phase_raw = np.angle(z)
    phase_fit = np.unwrap(np.concatenate([phase_raw[:n_win], phase_raw[-n_win:]]))

    # Fit lineare della fase
    p = np.polyfit(f_fit, phase_fit, 1)

    # Correggi la fase su tutto il range
    delay_phase = np.exp(-1j * np.polyval(p, f))
    z_corrected = z * delay_phase

    return z_corrected


from scipy.linalg import eig

def fit_circle(x, y):
    """
    Circle fit using the Chernov-Lesort method.
    Input:
        x, y : arrays of Re(S21), Im(S21)
    Returns:
        xc, yc : center of the circle
        r      : radius of the circle
    """
    w = x**2 + y**2

    # Compute the moments
    Mww = np.sum(w * w)
    Mxw = np.sum(x * w)
    Myw = np.sum(y * w)
    Mx  = np.sum(x)
    My  = np.sum(y)
    Mw  = np.sum(w)
    Mxx = np.sum(x * x)
    Myy = np.sum(y * y)
    Mxy = np.sum(x * y)
    n   = len(x)

    # Matrix M
    M = np.array([
        [Mww, Mxw, Myw, Mw],
        [Mxw, Mxx, Mxy, Mx],
        [Myw, Mxy, Myy, My],
        [Mw,  Mx,  My,  n ]
    ])

    # Constraint matrix B
    B = np.zeros((4, 4))
    B[1, 1] = 1
    B[2, 2] = 1

    # Solve generalized eigenvalue problem: M a = η B a
    eigvals, eigvecs = eig(M, B)

    # Filter real positive eigenvalues
    real_pos = [(val, vec) for val, vec in zip(eigvals, eigvecs.T) if np.isreal(val) and val > 0]
    if not real_pos:
        raise RuntimeError("No positive real eigenvalues found in circle fit.")

    eta, A = sorted(real_pos, key=lambda t: t[0])[0]
    A = np.real(A)  # Ensure it's real

    A_coeff, B_coeff, C_coeff, D_coeff = A

    # Extract center and radius
    xc = -B_coeff / (2 * A_coeff)
    yc = -C_coeff / (2 * A_coeff)
    r = np.sqrt((B_coeff**2 + C_coeff**2 - 4 * A_coeff * D_coeff) / (4 * A_coeff**2))

    return xc, yc, r

# -----------------------------------------------
# Traslazione e rotazione della circonferenza
# Rif. Appendice E.2.3
# -----------------------------------------------
def rotate_and_center(z, xc, yc):
    zc = xc + 1j * yc                   # Centro complesso
    angle = np.angle(zc)               # Angolo di rotazione
    return (z - zc) * np.exp(-1j * angle), angle  # Nuovo sistema di riferimento


# -----------------------------------------------
# Modello teorico per la fase del risonatore
# Rif. Appendice E.2.4 Eq. (E.11)
# -----------------------------------------------
def phase_model(f, fr, Qt, theta0):
    return theta0 + 2 * np.arctan(2 * Qt * (1 - f / fr))


# -----------------------------------------------
# Fit della fase usando Minuit
# Rif. Appendice E.2.4 - stima di fr, Qt e θ0
# -----------------------------------------------
def fit_phase_minuit(f, z_rot):
    phase = np.unwrap(np.angle(z_rot))         # Estrae la fase
    fr0 = f[np.argmin(np.abs(np.angle(z_rot)))]  # Stima iniziale di fr (minima fase)
    cost = LeastSquares(f, phase, np.ones_like(phase), phase_model)
    minuit = Minuit(cost, fr=fr0, Qt=10000, theta0=0)
    minuit.migrad()
    return minuit.values['fr'], minuit.values['Qt'], minuit.values['theta0']


# -----------------------------------------------
# Calcolo di Qc e Qi da Qt e geometria del cerchio
# Rif. Appendice E.2.5 Eq. (E.13)
# -----------------------------------------------
def calc_quality_factors(xc, yc, r, Qt):
    Qc = ((np.sqrt(xc**2 + yc**2) + r) / (2 * r)) * Qt
    Qi = 1 / (1/Qt - 1/Qc)
    return Qc, Qi


# -----------------------------------------------
# Pipeline completa per estrazione parametri del risonatore
# Unisce E.2.1 - E.2.5
# -----------------------------------------------
def fit_full_resonator(filename):
    # 1. Lettura e pre-processing
    f, z = read_data(filename)
    z_nodelay = remove_cable_delay(f, z)

    # 2. Fit del cerchio nel piano complesso
    xc, yc, r = fit_circle(z_nodelay.real, z_nodelay.imag)

    # 3. Rotazione del cerchio
    z_rot, angle = rotate_and_center(z_nodelay, xc, yc)

    # 4. Fit della fase per ottenere fr, Qt, theta0
    fr, Qt, theta0 = fit_phase_minuit(f, z_rot)

    # 5. Calcolo dei fattori di qualità
    Qc, Qi = calc_quality_factors(xc, yc, r, Qt)

    # --- Plot dei risultati ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot (a): cerchio complesso
    axs[0].plot(z.real, z.imag, 'o', color="#9edae5", markersize=1.5, label="Dati originali")
    axs[0].plot(z_nodelay.real, z_nodelay.imag, 'o', color="#1f77b4", markersize=2, label="Senza delay")
    circle = plt.Circle((xc, yc), r, color="#ff7f0e", fill=False, linewidth=2, label='Fit circolare')
    axs[0].add_artist(circle)
    axs[0].set_xlabel("Re[Z]")
    axs[0].set_ylabel("Im[Z]")
    axs[0].set_title("Circular fit")
    axs[0].axis('equal')
    axs[0].grid(True, linestyle="--", alpha=0.6)
    axs[0].legend(fontsize=8, loc="best")

    # Subplot (b): fase vs frequenza
    phase_data = np.unwrap(np.angle(z_rot))
    axs[1].plot(f, phase_data, '.', color="#2ca02c", markersize=2.5, label="Fase misurata")
    axs[1].plot(f, phase_model(f, fr, Qt, theta0), '-', color="#d62728", linewidth=2,
                label=f'Phase fit\n$Q_i = {Qi:.0f}$')
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Phase (rad)")
    axs[1].set_title("Resonator phase fit")
    axs[1].grid(True, linestyle="--", alpha=0.6)
    axs[1].legend(fontsize=9, loc="best")

    plt.suptitle("Resonator analysis", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return Qc, Qi, Qt, fr


# -----------------------------------------------
# Modello teorico per la variazione di 1/Qi con T
# basato su teoria di Mattis-Bardeen
# -----------------------------------------------
def model(T, fr, Q0, Delta0_meV, alpha=0.8):
    meV_to_J = 1.60218e-22               # Conversione meV → Joule
    Delta0 = Delta0_meV * meV_to_J
    T = np.asarray(T, dtype=float) * 1e-3  # Converte mK → K
    xi = (hbar * fr) / (2 * k * T)

    # Termini della funzione di risposta
    bessel_term = np.sinh(xi) * kv(0, xi)
    iv_term = np.exp(-xi) * iv(0, -xi)

    num = np.exp(-Delta0 / (k * T)) * bessel_term
    den = 1 - 2 * np.exp(-Delta0 / (k * T)) * iv_term

    return (1/Q0 + (2 * alpha / np.pi) * (num / den))


# -----------------------------------------------
# Fit del gap Δ0 tramite la dipendenza termica di Qi(T)
# usando modello teorico (Mattis-Bardeen)
# -----------------------------------------------
def fit_delta(file_list, Qi, T, freq, alpha_fixed=0.8):
    f_min_cut=3.1765e9
    T = np.asarray(T, dtype=float)
    Qi = np.asarray(Qi, dtype=float)
    Qi_err = 0.01 * Qi                       # 1% errore relativo
    inv_Qi_err = Qi_err / Qi**2              # Propagazione errore su 1/Qi


     # Mappa colori per rappresentare la temperatura
    norm = plt.Normalize(T.min(), T.max())
    cmap = cm.get_cmap("coolwarm")
    colors = [cmap(norm(t)) for t in T]

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    # Traccia le risonanze in ampiezza per ogni file
    for file, color in zip(file_list, colors):
        f, z = read_data(file)
        mask = f >= f_min_cut
        axs[0].plot(f[mask], np.abs(z)[mask], color=color, linewidth=1)
    
    for t, df, color in zip(T, 1/Qi, colors):
        axs[1].plot(t, df, 'o', color=color)

    axs[0].set_xlabel("Frequency [Hz]")
    axs[0].set_ylabel(r"Amplitude $|S_{21}|$")
    axs[0].set_title("Resonances behavior at different temperatures")
    axs[0].grid(True, linestyle="--", alpha=0.5)

        # Fit con Minuit
    least_squares = LeastSquares(T, 1/Qi, inv_Qi_err, model)
    minuit = Minuit(least_squares,
                    fr=freq,
                    Q0=max(Qi),
                    Delta0_meV=0.3,
                    alpha=alpha_fixed)

    minuit.limits["Q0"] = (1e3, None)
    minuit.limits["Delta0_meV"] = (0.1, 0.5)
    minuit.fixed["fr"] = True
    minuit.fixed["alpha"] = True

    minuit.migrad()
    minuit.hesse()

    # Stampa dei risultati
    print("Successo del fit:", minuit.valid)
    print("Chi-quadro ridotto:", minuit.fval / minuit.ndof)
    for par, val, err in zip(minuit.parameters, minuit.values, minuit.errors):
        print(f"{par} = {val:.4f} ± {err:.4f}")

    Q0_fit = minuit.values["Q0"]
    Delta0_fit = minuit.values["Delta0_meV"]


    # Traccia 1/qi rispetto alla temperatura
    for t, df, color in zip(T, 1/Qi, colors):
        axs[1].plot(t, df, 'o', color=color)

    # Plot del fit teorico
    axs[1].plot(T, model(T, freq, Q0_fit, Delta0_fit), '-', color="tab:blue", linewidth=2,
                label=fr"Fit MB: $\Delta_0$ = {Delta0_fit:.3f} meV")

    axs[1].set_xlabel("Temperatures (mK)")
    axs[1].set_ylabel(r"1/$Q_i$")
    axs[1].set_title("1/Qi vs Temperature")
    axs[1].grid(True, linestyle="--", alpha=0.5)
    axs[1].legend(fontsize=9)

    plt.tight_layout()
    plt.show()

    return Q0_fit, Delta0_fit


