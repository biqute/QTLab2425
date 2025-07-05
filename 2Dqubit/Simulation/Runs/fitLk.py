import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from matplotlib.gridspec import GridSpec
import scipy.optimize as spopt
import csv
import re

# -----------------------------------------------
# Lettura dei dati CSV organizzati per blocchi Lk
# -----------------------------------------------
def lettura_csv(nome_file):
    """
    Legge un file CSV contenente blocchi di dati con intestazione 'Lk = ...'.
    Restituisce due dizionari ordinati contenenti i dati per ciascun Lk.
    """
    dati_col1 = {}
    dati_col2 = {}

    lk_corrente = None
    colonna1 = []
    colonna2 = []

    def è_riga_numerica(riga):
        if len(riga) != 2:
            return False
        try:
            float(riga[0])
            float(riga[1])
            return True
        except ValueError:
            return False

    with open(nome_file, newline='') as csvfile:
        lettore = csv.reader(csvfile)
        for riga in lettore:
            riga = [elem.strip() for elem in riga]
            if len(riga) == 1 and "Lk=" in riga[0]:
                if lk_corrente is not None and colonna1 and colonna2:
                    dati_col1[lk_corrente] = colonna1
                    dati_col2[lk_corrente] = colonna2
                    colonna1 = []
                    colonna2 = []
                match = re.search(r"Lk\s*=\s*([0-9.]+)", riga[0])
                lk_corrente = match.group(1) if match else "Unknown"
            elif è_riga_numerica(riga) and lk_corrente is not None:
                colonna1.append(float(riga[0]))
                colonna2.append(float(riga[1]))

        if lk_corrente is not None and colonna1 and colonna2:
            dati_col1[lk_corrente] = colonna1
            dati_col2[lk_corrente] = colonna2

        dati_col1 = dict(sorted(dati_col1.items(), key=lambda item: float(item[0])))
        dati_col2 = dict(sorted(dati_col2.items(), key=lambda item: float(item[0])))

    return dati_col1, dati_col2

# -----------------------------------------------
# Visualizzazione e stampa dei dati per ciascun Lk
# -----------------------------------------------
def stampa(dati_col1, dati_col2):
    """
    Stampa i dati per ciascun Lk e disegna i grafici delle risposte in frequenza.
    """
    for lk_val in dati_col1:
        print(f"\nDati per Lk={lk_val}")
        print(f"colonna1_{lk_val}Lk =", dati_col1[lk_val])
        print(f"colonna2_{lk_val}Lk =", dati_col2[lk_val])

    plt.figure(figsize=(10, 6))
    for lk_val in dati_col1:
        x = dati_col1[lk_val]
        y = dati_col2[lk_val]
        plt.plot(x, y, label=f"Lk={lk_val}")
    plt.xlabel("Frequenza (GHz)")
    plt.ylabel("dB[S21]")
    plt.title("Risposte in frequenza per diversi Lk")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------
# Estrazione dei minimi di trasmissione per ogni Lk
# -----------------------------------------------
def trova_minimi(dati_col1, dati_col2, discarded_Lk=None):
    """
    Estrae i minimi delle curve S21 per ogni Lk (escludendo eventuali scartati).
    """
    if discarded_Lk is None:
        discarded_Lk = []

    lk_list = []
    fr_list = []

    for lk_str in dati_col1:
        try:
            lk_val = float(lk_str)
        except ValueError:
            continue

        if lk_val not in discarded_Lk:
            x = np.array(dati_col1[lk_str]) 
            y = np.array(dati_col2[lk_str])  
            lk_list.append(lk_val)

            # Stima iniziale della frequenza di risonanza (minimo)
            i = np.argmin(y)
            fmin = x[i]
            amp_min = np.min(y)
            amp_max = np.max(y)

            # Stima approssimata per Qt
            tolleranza = 0.2e-1
            amp_FWHM = amp_min + (amp_max - amp_min) / 2
            indici = np.where(np.abs(y - amp_FWHM) < tolleranza)[0]
            frequenze_half = x[indici]

            Qc_guess = 14000
            Qt_guess = fmin / np.abs(frequenze_half[0] - frequenze_half[-1]) if len(frequenze_half) >= 2 else 15000
            k_guess = (amp_max - amp_min) * (Qc_guess / Qt_guess)
            amp_err = np.ones(len(y)) * 1e-3  # Errore costante

            # Parametri iniziali del modello
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
                x_ = f - fr
                resonant = k * np.abs(1 - Qt * np.exp(1j * phi) / Qc / (1 + 2j * Qt * x_ / fr))
                return a + b * x_ + c * x_**2 + d * x_**3 + resonant

            # Fit con Minuit
            least_squares = LeastSquares(x, y, amp_err, model)
            minuit = Minuit(least_squares, **initial_params)

            minuit.limits["k"] = (0.5, 5)
            minuit.limits["phi"] = (-np.pi, np.pi)
            minuit.limits["Qt"] = (500, 50000)
            minuit.limits["delta_Q"] = (10, 50000)
            minuit.limits["fr"] = (fmin - 5e4, fmin + 5e4)

            minuit.migrad()
            minuit.hesse()

            fr_fit = minuit.values['fr']
            fr_list.append(fr_fit)

    return fr_list, lk_list

# -----------------------------------------------
# Visualizzazione semplice dei minimi fr vs Lk
# -----------------------------------------------
def lkvfr (Lk, f_min):
    """
    Plotta la frequenza di risonanza minima in funzione di Lk.
    """
    plt.plot(Lk, f_min, marker='o', linestyle='', color='b', label='Minimi')
    plt.ylabel("Frequenza (GHz)")
    plt.xlabel("Lk")
    plt.title("Minimi della risposta in frequenza per Lk selezionati")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------
# Modello teorico: fr(Lk) = a / sqrt(Lk + g) + b
# -----------------------------------------------
def model (Lk, a, b, g):
    """
    Modello del comportamento di fr in funzione di Lk.
    """
    Lk = np.array(Lk)
    return ((a / np.sqrt(Lk + g)) + b)

# -----------------------------------------------
# Fit del modello ai dati
# -----------------------------------------------
def fit (x, y):
    """
    Fit del modello teorico fr(Lk) ai dati sperimentali.
    """
    x_err = np.ones(len(x)) * 0.1
    y_err = np.ones(len(y)) * 0.1
    least_squares = LeastSquares(x, y, y_err, model)
    minuit = Minuit(least_squares, a=1, b=0, g=0.1)
    minuit.migrad()
    minuit.hesse()

    print("Successo del fit:", minuit.valid)
    print("Chi-quadro ridotto:", minuit.fval / minuit.ndof)
    for par, val, err in zip(minuit.parameters, minuit.values, minuit.errors):
        print(f"{par} = {val:.3f} ± {err:.3f}")

    a_fit = minuit.values["a"]
    b_fit = minuit.values["b"]
    g_fit = minuit.values["g"]

    fig, ax = plt.subplots()
    ax.plot(x, y, "bo", label="Dati sperimentali", marker="o", markersize=2, linestyle="none")
    ax.plot(x, model(x, a_fit, b_fit, g_fit), "r-", label="Fit")
    ax.set_xlabel("Lk")
    ax.set_ylabel("Frequenza (GHz)")
    ax.legend()
    plt.title("Fit della frequenza minima in funzione di Lk")
    plt.show()

    return a_fit, b_fit, g_fit

# -----------------------------------------------
# Inversione della formula: da fr a Lk
# -----------------------------------------------
def calcolo_lk (a, b, g, fr):
    """
    Calcola Lk dato fr invertendo il modello.
    """
    Lk = (a / (fr - b))**2 - g
    return Lk

# -----------------------------------------------
# Fit multiplo e confronto con Lk medio
# -----------------------------------------------
def lkvfr_tot(Lk1, f_min1, Lk2, f_min2, Lk3, f_min3, Lk4, f_min4, lk_mean, lk_locali):
    """
    Esegue il fit su 4 set di dati fr vs Lk, confrontando ciascuno con il valore medio lk_mean.
    """
    lk_array = [np.array(Lk1), np.array(Lk2), np.array(Lk3), np.array(Lk4)]
    f_min_array = [np.array(f_min1), np.array(f_min2), np.array(f_min3), np.array(f_min4)]

    cmap = plt.get_cmap("plasma")
    colori = [cmap(i) for i in np.linspace(0.2, 0.8, 4)]

    plt.figure(figsize=(10, 6))

    for i in range(4):
        x = lk_array[i]
        y = f_min_array[i]
        col = colori[i]
        lk_local = lk_locali[i]

        y_err = np.ones(len(y)) * 0.05
        least_squares = LeastSquares(x, y, y_err, model)
        minuit = Minuit(least_squares, a=1, b=0, g=0.1)
        minuit.migrad()
        minuit.hesse()

        a_fit = minuit.values["a"]
        b_fit = minuit.values["b"]
        g_fit = minuit.values["g"]

        fr_local = model(lk_local, a_fit, b_fit, g_fit)
        fr_at_lk_mean = model(lk_mean, a_fit, b_fit, g_fit)

        print(f"Resonator {i+1}:")
        print(f"   - Lk locale fornito = {lk_local:.3f}")
        print(f"   - f_res(Lk locale) = {fr_local} GHz")
        print(f"   - f_res(Lk_mean)   = {fr_at_lk_mean} GHz")
        print(f" delta fr = {np.abs(fr_local - fr_at_lk_mean)}")

        x_fit = np.linspace(min(x), max(x), 300)
        y_fit = model(x_fit, a_fit, b_fit, g_fit)

        label_fit = f"Resonator {i+1} (L_k = {lk_local:.2f})"
        plt.plot(x, y, 'o', color=col, label=f'Data {i+1}')
        plt.plot(x_fit, y_fit, '-', color=col, label=label_fit)
        plt.plot(lk_mean, fr_at_lk_mean, 'ro', markersize=6)

    plt.axvline(x=lk_mean, color='black', linestyle='--', linewidth=1.2, label=f'Lk mean = {lk_mean:.2f}')
    plt.xlabel("Lk")
    plt.ylabel("Frequency (GHz)")
    plt.title("Resonance Frequencies vs Lk and compatibility with mean Lk")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------
# Taglio dei dati in un intervallo definito
# -----------------------------------------------
def cut(dati_col1, dati_col2, min_x, max_x):
    """
    Restituisce solo i dati compresi tra min_x e max_x.
    """
    nuovi_col1 = {}
    nuovi_col2 = {}

    for lk in dati_col1:
        x_vals = dati_col1[lk]
        y_vals = dati_col2[lk]

        if len(x_vals) != len(y_vals):
            raise ValueError(f"x_vals e y_vals hanno lunghezze diverse per Lk={lk}")

        x_filtrati = []
        y_filtrati = []

        for x, y in zip(x_vals, y_vals):
            if min_x <= x <= max_x:
                x_filtrati.append(x)
                y_filtrati.append(y)

        nuovi_col1[lk] = x_filtrati
        nuovi_col2[lk] = y_filtrati

    return nuovi_col1, nuovi_col2

# -----------------------------------------------
# Taglio selettivo solo su Lk specifici
# -----------------------------------------------
def selected_cut(dati_col1, dati_col2, lk_target, min_x, max_x):
    """
    Applica un filtro di intervallo [min_x, max_x] solo ai Lk specificati.
    """
    if not isinstance(lk_target, (list, set, tuple)):
        lk_target = [lk_target]

    lk_target = {str(lk) for lk in lk_target}

    nuovi_col1 = {}
    nuovi_col2 = {}

    for lk in dati_col1:
        x_vals = dati_col1[lk]
        y_vals = dati_col2[lk]

        if lk in lk_target:
            x_filtrati = []
            y_filtrati = []
            for x, y in zip(x_vals, y_vals):
                if min_x <= x <= max_x:
                    x_filtrati.append(x)
                    y_filtrati.append(y)
            nuovi_col1[lk] = x_filtrati
            nuovi_col2[lk] = y_filtrati
        else:
            nuovi_col1[lk] = x_vals
            nuovi_col2[lk] = y_vals

    return nuovi_col1, nuovi_col2
