import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.special import iv, kv
from scipy.constants import hbar, k

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
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    from iminuit import Minuit
    from iminuit.cost import LeastSquares
    from scipy.special import iv, kv
    from scipy.constants import hbar, k

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
    T_min = 200
    T_max = 350
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
    m.fixed["alpha"] = True  # α fissato
    m.limits["Delta0_meV"] = (0.1, 0.5)
    m.migrad()
    m.hesse()

    delta_fit = m.values["Delta0_meV"]  # Valore finale di Δ₀

    # Plot del fit teorico
    T_vis = np.linspace(T_min, T_max, 300)
    axs[1].plot(T_vis, model(T_vis, 0.8, delta_fit), '-', color="tab:blue", linewidth=2,
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
    print(f"  α = 0.8 (fissato)")
    print(f"  Δ₀ = {delta_fit:.4f} meV")
