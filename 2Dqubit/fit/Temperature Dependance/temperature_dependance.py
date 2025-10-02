import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.special import iv, kv
from scipy.constants import hbar, k
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
# Legge i dati da file CSV (colonne: frequenza, I, Q)
def read_data(filename):
    data = np.loadtxt(filename, delimiter=",")
    f = data[:, 0]
    I = data[:, 1]
    Q = data[:, 2]
    z = I + 1j * Q  # Costruisce il segnale complesso
    return f, z

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

# Trasla e ruota la traiettoria complessa in modo da centrare e allineare il cerchio
def rotate_and_center(z, xc, yc):
    zc = xc + 1j * yc
    angle = np.angle(zc)
    return (z - zc) * np.exp(-1j * angle), angle

# Modello della fase del risonatore vicino alla risonanza
def phase_model(f, fr, Qt, theta0):
    return theta0 + 2 * np.arctan(2 * Qt * (1 - f / fr))

# Fit della fase con Minuit usando il modello sopra
def fit_phase_minuit(f, z_rot, fr_guess=None, Qt_guess=10000, theta0_guess=0):
    phase = np.unwrap(np.angle(z_rot))

    if fr_guess is None:
        fr_guess = f[np.argmin(np.abs(np.angle(z_rot)))]  # Stima iniziale del fr

    cost = LeastSquares(f, phase, np.ones_like(phase), phase_model)  # Cost function
    minuit = Minuit(cost, fr=fr_guess, Qt=Qt_guess, theta0=theta0_guess)
    minuit.migrad()  # Minimizzazione
    return minuit.values['fr'], minuit.values['Qt'], minuit.values['theta0']

# Calcola Qc e Qi a partire dai parametri geometrici del fit circolare
def calc_quality_factors(xc, yc, r, Qt):
    Qc = ((np.sqrt(xc**2 + yc**2) + r) / (2 * r)) * Qt  # Coupling Q
    Qi = 1 / (1/Qt - 1/Qc)  # Internal Q
    return Qc, Qi

# Funzione completa per analizzare un singolo file di misura e plottare i risultati
def fit_full_resonator(filename, fr_guess=None, Qt_guess=10000, theta0_guess=0):
    f, z = read_data(filename)
    z_nodelay = remove_cable_delay(f, z)

    xc, yc, r = fit_circle(z_nodelay.real, z_nodelay.imag)  # Fit circolare
    z_rot, angle = rotate_and_center(z_nodelay, xc, yc)  # Traslazione e rotazione

    fr, Qt, theta0 = fit_phase_minuit(f, z_rot, fr_guess, Qt_guess, theta0_guess)  # Fit della fase
    Qc, Qi = calc_quality_factors(xc, yc, r, Qt)  # Calcolo dei fattori di qualità

    # Grafico dei dati complessi e della fase
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(z.real, z.imag, 'o', color="#9edae5", markersize=1.5, label="Dati originali")
    axs[0].plot(z_nodelay.real, z_nodelay.imag, 'o', color="#1f77b4", markersize=2, label="Senza delay")
    circle = plt.Circle((xc, yc), r, color="#ff7f0e", fill=False, linewidth=2, label='Fit circolare')
    axs[0].add_artist(circle)
    axs[0].set_xlabel("Re[Z]")
    axs[0].set_ylabel("Im[Z]")
    axs[0].set_title("Fit circolare")
    axs[0].axis('equal')
    axs[0].grid(True, linestyle="--", alpha=0.6)
    axs[0].legend(fontsize=8, loc="best")

    phase_data = np.unwrap(np.angle(z_rot))
    axs[1].plot(f, phase_data, '.', color="#2ca02c", markersize=2.5, label="Fase misurata")
    axs[1].plot(f, phase_model(f, fr, Qt, theta0), '-', color="#d62728", linewidth=2,
                label=f'Fit fase\n$Q_i = {Qi:.0f}$')
    axs[1].set_xlabel("Frequenza (Hz)")
    axs[1].set_ylabel("Fase (rad)")
    axs[1].set_title("Fit della fase del risonatore")
    axs[1].grid(True, linestyle="--", alpha=0.6)
    axs[1].legend(fontsize=9, loc="best")

    plt.suptitle("Analisi del risonatore con fit estetici", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return Qc, Qi, Qt, fr  # Restituisce i parametri principali del risonatore

# Funzione per analizzare la variazione della frequenza di risonanza in funzione della temperatura
def plot_resonances(file_list, temperatures, fr_list, f_min_cut=3.1765e9):

    # Modello teorico basato sulla teoria di Mattis-Bardeen per la variazione della frequenza
   
    def model(T_mK, alpha, Delta0_meV):
            T = np.asarray(T_mK) * 1e-3  # Conversione da mK a K
            Delta0 = Delta0_meV * 1.60218e-22  # Conversione da meV a joule
            prefactor = (alpha / 2) * np.sqrt((2 * np.pi * Delta0) / (k * T))
        
            return - prefactor * np.exp(- Delta0 / (k * T))
   

    T = np.asarray(temperatures)
    fr = np.asarray(fr_list)
    f0 = np.max(fr)  # Frequenza di riferimento
    df_over_f0 = (fr - f0) / f0  # Δf / f0

    # Maschera per il range di temperatura del fit
    T_min = 30
    T_max = 420
    mask_fit = (T >= T_min) & (T <= T_max)
    Temp_cut = T[mask_fit]
    df_cut = df_over_f0[mask_fit]

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

    axs[0].set_xlabel("Frequenza [Hz]")
    axs[0].set_ylabel(r"Ampiezza $|S_{21}|$")
    axs[0].set_title("Risonanze a varie temperature")
    axs[0].grid(True, linestyle="--", alpha=0.5)

    # Traccia df/f0 rispetto alla temperatura
    for t, df, color in zip(T, df_over_f0, colors):
        axs[1].plot(t, df, 'o', color=color)

    # Fit del modello MB ai dati
    df_err = 0.05 * np.abs(df_cut) + 1e-8   # Errore fittizio uniforme
    least_squares = LeastSquares(Temp_cut, df_cut, df_err, model)
    m = Minuit(least_squares, alpha=0.8, Delta0_meV=0.5)  # Fit iniziale
    m.limits["alpha"] = (0, 5.0)
    m.limits["Delta0_meV"] = (0, 0.5)
    m.migrad()
    m.hesse()

    delta_fit = m.values["Delta0_meV"]  # Valore finale di delta
    alpha_fit = m.values["alpha"]  # Valore finale di alpha

    # Plot del fit teorico
    T_vis = np.linspace(T_min, T_max, 300)
    axs[1].plot(T_vis, model(T_vis, alpha_fit , delta_fit), '-', color="tab:blue", linewidth=2,
                label=fr"Fit MB: $\Delta_0$ = {delta_fit:.3f} meV")

    axs[1].set_xlabel("Temperatura (mK)")
    axs[1].set_ylabel(r"$\Delta f / f_0$")
    axs[1].set_title("Variazione relativa della frequenza")
    axs[1].grid(True, linestyle="--", alpha=0.5)
    axs[1].legend(fontsize=9)

    plt.tight_layout()
    plt.show()

    # Stampa del risultato finale del fit
    print("Fit completato:")
    print(f"  α = {alpha_fit:.4f}")
    print(f"  Δ₀ = {delta_fit:.4f} meV")


#fit lorenziano per confronto
def read_data_lo(file_path, Del=None):
    """
    Legge i dati dal file. Supporta:
    - formato con 2 colonne: (frequenza, ampiezza)
    - formato con 3 colonne: (frequenza, I, Q), e calcola ampiezza = sqrt(I² + Q²)
    L'output è normalizzato tra 0 e 1.
    """
    data = np.loadtxt(file_path, delimiter=Del)
    num_columns = data.shape[1]
    f = data[:, 0]

    if num_columns == 3:
        I = data[:, 1]
        Q = data[:, 2]
        amp = np.sqrt(I**2 + Q**2)
        amp = amp / np.max(amp)
    else:
        amp = data[:, 1]

    return f, amp


def fit_cut(file_path, unit=None, Del=None, cut_min=None, cut_max=None,
            data_min=None, data_max=None, initial_params=None, show="yes"):
   

    # Caricamento dati
    f, amp = read_data_lo(file_path, Del)

    # Taglio dati visibili (per plottare ampiezza grezza, opzionale)
    if data_min is not None and data_max is not None:
        if data_min >= data_max:
            raise ValueError("data_min deve essere minore di data_max")
        mask = (f >= data_min) & (f <= data_max)
        f_cut_data = f[mask]
        amp_cut_data = amp[mask]
    else:
        f_cut_data = f
        amp_cut_data = amp

    # Taglio effettivo dei dati da fittare
    if cut_min is not None and cut_max is not None:
        if cut_min >= cut_max:
            raise ValueError("cut_min deve essere minore di cut_max")
        mask = (f >= cut_min) & (f <= cut_max)
        f_cut = f[mask]
        amp_cut = amp[mask]
    else:
        f_cut = f
        amp_cut = amp

    # Stima iniziale della frequenza di risonanza (minimo)
    i = np.argmin(amp_cut)
    fmin = f_cut[i]
    amp_min = np.min(amp_cut)
    amp_max = np.max(amp_cut)

    # Stima approssimata per Qt
    tolleranza = 0.2e-1
    amp_FWHM = amp_min + (amp_max - amp_min) / 2
    indici = np.where(np.abs(amp_cut - amp_FWHM) < tolleranza)[0]
    frequenze_half = f_cut[indici]

    Qc_guess = 14000
    Qt_guess = fmin / np.abs(frequenze_half[0] - frequenze_half[-1]) if len(frequenze_half) >= 2 else 15000
    k_guess = (amp_max - amp_min) * (Qc_guess / Qt_guess)
    amp_err = np.ones(len(amp_cut)) * 1e-3  # Errore costante

    # Parametri iniziali del modello
    if initial_params is None:
        initial_params = {
            'a': 1,
            'b': 1e-9,
            'c': 1e-18,
            'd': 1e-27,
            'k': k_guess,
            'phi': 0.01,
            'Qt': Qt_guess,
            'delta_Q': 1000,
            'fr': fmin
        }

    # Modello teorico: polinomio + risposta risonante (complessa)
    def model(f, a, b, c, d, k, phi, Qt, delta_Q, fr):
        Qc = Qt + np.abs(delta_Q)
        x = f - fr
        resonant = k * np.abs(1 - Qt * np.exp(1j * phi) / Qc / (1 + 2j * Qt * x / fr))
        return a + b * x + c * x**2 + d * x**3 + resonant

    # Fit con Minuit
    least_squares = LeastSquares(f_cut, amp_cut, amp_err, model)
    minuit = Minuit(least_squares, **initial_params)

    # Limiti sui parametri fisici
    minuit.limits["k"] = (0.5, 5)
    minuit.limits["phi"] = (-np.pi, np.pi)
    minuit.limits["Qt"] = (500, 50000)
    minuit.limits["delta_Q"] = (10, 50000)
    minuit.limits["fr"] = (fmin - 5e4, fmin + 5e4)

    # Ottimizzazione
    minuit.migrad()
    minuit.hesse()

    # Estrazione parametri fit
    a_fit = minuit.values['a']
    b_fit = minuit.values['b']
    c_fit = minuit.values['c']
    d_fit = minuit.values['d']
    k_fit = minuit.values['k']
    phi_fit = minuit.values['phi']
    Qt_fit = minuit.values['Qt']
    delta_Q_fit = minuit.values['delta_Q']
    Qc_fit = Qt_fit + abs(delta_Q_fit)
    fr_fit = minuit.values['fr']

    # Calcolo modello e residui
    amp_fit = model(f_cut, a_fit, b_fit, c_fit, d_fit, k_fit, phi_fit, Qt_fit, delta_Q_fit, fr_fit)
    residui = (amp_cut - amp_fit) / amp_cut

    # Plot risultati
    if show == "yes":
        print('Fit riuscito:', minuit.valid)
        print('Chi quadro ridotto:', minuit.fval / minuit.ndof)
        print(f'fr = {fr_fit:.3f} Hz, Qt = {Qt_fit:.2f}, Qc = {Qc_fit:.2f}')
        for par, val, err in zip(minuit.parameters, minuit.values, minuit.errors):
            print(f'{par} = {val:.3e} ± {err:.3e}')

        fig = plt.figure(figsize=(10, 6))
        gs = GridSpec(2, 1, height_ratios=[3, 1.5], hspace=0.3)

        # Parte superiore: fit dati
        ax_fit = fig.add_subplot(gs[0])
        ax_fit.plot(f_cut_data, amp_cut_data, label="Data", marker="o", markersize=3, linestyle="none", alpha=0.7)
        ax_fit.plot(f_cut, amp_fit, label=f"Fit (fr = {fr_fit/1e6:.6f} MHz)", color="darkred", linewidth=2)
        ax_fit.set_title("Resonance Fit", fontsize=14)
        ax_fit.set_xlabel("Frequency (Hz)", fontsize=12)
        ax_fit.set_ylabel("Amplitude", fontsize=12)
        ax_fit.set_yscale("log")
        ax_fit.legend(fontsize=10)
        ax_fit.grid(True, linestyle='--', alpha=0.6)

        # Parte inferiore: residui
        ax_res = fig.add_subplot(gs[1])
        ax_res.plot(f_cut, residui, label="Residuals", marker="o", markersize=2.5, linestyle="none", color="steelblue")
        ax_res.axhline(0, color='black', linestyle="--", linewidth=1)
        ax_res.set_xlabel("Frequency (Hz)", fontsize=12)
        ax_res.set_ylabel("Residuals", fontsize=12)
        ax_res.legend(fontsize=10)
        ax_res.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()

    # Restituzione dei parametri ottenuti
    return fr_fit, Qc_fit, Qt_fit, a_fit, b_fit, c_fit, d_fit, k_fit, phi_fit
