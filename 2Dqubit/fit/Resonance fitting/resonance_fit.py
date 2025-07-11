import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from matplotlib.gridspec import GridSpec
import scipy.optimize as spopt
from scipy import stats
import re

# -----------------------------------------------
# Lettura dati da file CSV o TXT
# -----------------------------------------------
def read_data(file_path, Del=None):
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

# -----------------------------------------------
# Conversione ampiezza: lineare <-> logaritmica
# -----------------------------------------------
def converter(amp, unit=None):
    """
    Converte i dati di ampiezza tra unità logaritmica e lineare:
    - "log"    : da lineare a dB normalizzati
    - "linear" : da dB a lineare normalizzato
    """
    if unit == "log":
        amp2 = 20 * np.log10(amp)
        return amp2 / np.max(amp2)
    elif unit == "linear":
        amp2 = 10**(amp / 20)
        return amp2 / np.max(amp2)
    else:
        return amp

# -----------------------------------------------
# Estrazione temperatura dal nome del file
# -----------------------------------------------
def extract_temperature(filename):
    """
    Estrae la temperatura da un nome file che contiene 'XXXmK' o 'X.XK'.
    Restituisce la temperatura in mK.
    """
    match = re.search(r'(\d+(?:\.\d+)?)(mK|K)', filename)
    if match:
        temp = float(match.group(1))
        unit = match.group(2)
        return temp if unit == "mK" else temp * 1000
    else:
        raise ValueError(f"Temperatura non trovata nel nome del file: {filename}")

# -----------------------------------------------
# Estrae temperature da una lista di file
# -----------------------------------------------
def process_files(file_list):
    """
    Estrae le temperature da una lista di nomi di file.
    Restituisce una lista di temperature in millikelvin.
    """
    temperatures = []
    for filename in file_list:
        try:
            temperature = extract_temperature(filename)
            temperatures.append(temperature)
        except ValueError as e:
            print(e)
    return temperatures

# -----------------------------------------------
# Fit della risonanza complessa (modulo ampiezza)
# con modello fenomenologico avanzato
# -----------------------------------------------
def fit_cut(file_path, unit=None, Del=None, cut_min=None, cut_max=None,
            data_min=None, data_max=None, initial_params=None, show="yes"):
    """
    Esegue il fit della curva di risonanza su ampiezza (modulo S21).
    - Applica tagli selettivi ai dati (data_cut, fit_cut)
    - Usa un modello che include fondo polinomiale + risposta risonante
    - Restituisce fr, Qc, Qt, e parametri del fit
    """

    # Caricamento dati
    f, amp = read_data(file_path, Del)

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
