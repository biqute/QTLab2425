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

# Rimuove il ritardo di fase dovuto ai cavi usando un fit lineare della fase
def remove_cable_delay(f, z):
    phase = np.unwrap(np.angle(z))
    p = np.polyfit(f, phase, 1)  # Fit lineare della fase
    delay_phase = np.exp(-1j * np.polyval(p, f))  # Compensazione del ritardo
    return z * delay_phase

# Fit circolare nel piano complesso per trovare il centro e il raggio della traiettoria del risonatore
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
    r = np.mean(np.sqrt((x - xc)**2 + (y - yc)**2))  # Calcolo del raggio medio
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
    def make_df_model_mb(f0):
        def model(T_mK, alpha, Delta0_meV):
            T = np.asarray(T_mK) * 1e-3  # Conversione da mK a K
            Delta0 = Delta0_meV * 1.60218e-22  # Conversione da meV a joule
            xi = (hbar * f0) / (2 * k * T)  # Argomento per funzioni speciali
            bessel_term = np.sinh(xi) * kv(0, xi)  # Parte del numeratore
            iv_term = np.exp(-xi) * iv(0, -xi)  # Parte del denominatore

            num = np.exp(-Delta0 / (k * T)) * bessel_term
            den = 1 - 2 * np.exp(-Delta0 / (k * T)) * iv_term

            return -(alpha / 2) * (num / den)  # Δf/f0
        return model

    T = np.asarray(temperatures)
    fr = np.asarray(fr_list)
    f0 = np.max(fr)  # Frequenza di riferimento
    df_over_f0 = (fr - f0) / f0  # Δf / f0

    # Maschera per il range di temperatura del fit
    T_min = 100
    T_max = 410
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
    model = make_df_model_mb(f0)
    df_err = np.full_like(df_cut, 0.05 * np.abs(df_cut).max())  # Errore fittizio uniforme
    least_squares = LeastSquares(Temp_cut, df_cut, df_err, model)
    m = Minuit(least_squares, alpha=0.8, Delta0_meV=0.3)  # Fit iniziale
    m.fixed["alpha"] = True  # α fissato
    m.limits["Delta0_meV"] = (0.1, 0.4)
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
