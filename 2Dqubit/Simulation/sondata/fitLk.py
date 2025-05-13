import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from matplotlib.gridspec import GridSpec
import scipy.optimize as spopt
import csv
import re
def lettura_csv(nome_file):

    dati_col1 = {}
    dati_col2 = {}

    # Variabili temporanee
    lk_corrente = None
    colonna1 = []
    colonna2 = []

    # Funzione per verificare se una riga è numerica
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

            # Cerca Lk in righe descrittive
            if len(riga) == 1 and "Lk=" in riga[0]:
                # Se stavamo accumulando dati, salviamo il blocco precedente
                if lk_corrente is not None and colonna1 and colonna2:
                    dati_col1[lk_corrente] = colonna1
                    dati_col2[lk_corrente] = colonna2
                    colonna1 = []
                    colonna2 = []

                # Estrai il valore di Lk
                match = re.search(r"Lk\s*=\s*([0-9.]+)", riga[0])
                if match:
                    lk_corrente = match.group(1)
                else:
                    lk_corrente = "Unknown"

            elif è_riga_numerica(riga) and lk_corrente is not None:
                colonna1.append(float(riga[0]))
                colonna2.append(float(riga[1]))

        # Salva anche l'ultimo blocco se presente
        if lk_corrente is not None and colonna1 and colonna2:
            dati_col1[lk_corrente] = colonna1
            dati_col2[lk_corrente] = colonna2

        dati_col1 = dict(sorted(dati_col1.items(), key=lambda item: float(item[0])))
        dati_col2 = dict(sorted(dati_col2.items(), key=lambda item: float(item[0])))

        # Mostra i risultati
        for lk_val in dati_col1:
            print(f"\n🟢 Dati per Lk={lk_val}")
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

    return dati_col1, dati_col2

def trova_minimi(dati_col1, dati_col2, min_lk=0.0, max_lk=14.5):
    import numpy as np

    f_min = []
    Lk_list = []

    for lk_str in dati_col1:
        try:
            lk_val = float(lk_str)
        except ValueError:
            continue

        if min_lk <= lk_val <= max_lk:
            x = dati_col1[lk_str]
            y = dati_col2[lk_str]

            if len(y) == 0:
                continue

            min_idx = np.argmin(y)
            f_min.append(x[min_idx])
            Lk_list.append(lk_val)
            

    return f_min, Lk_list

def lkvfr (Lk, f_min) :

    plt.plot(Lk, f_min, marker='o', linestyle='', color='b', label='Minimi')
    plt.ylabel("Frequenza (GHz)")
    plt.xlabel("Lk")
    plt.title("Minimi della risposta in frequenza per Lk selezionati")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def model (Lk, a, b, g):

    Lk = np.array(Lk)
    
    return ((a/(np.sqrt(Lk+g)))+b)

def fit (x, y):

    x_err = np.ones(len(x)) * 0.1
    y_err = np.ones(len(y)) * 0.1
    
    least_squares = LeastSquares(x, y, y_err, model)
    minuit = Minuit(least_squares,
                    a=1,
                    b=0,
                    g=0.1)
    
    # Esegui il fit
    minuit.migrad()
    minuit.hesse()

    # Stampa risultati
    print("Successo del fit:", minuit.valid)
    print("Chi-quadro ridotto:", minuit.fval / minuit.ndof)
    for par, val, err in zip(minuit.parameters, minuit.values, minuit.errors):
        print(f"{par} = {val:.3f} ± {err:.3f}")

    # Estrai parametri
    a_fit = minuit.values["a"] 
    b_fit = minuit.values["b"]
    g_fit = minuit.values["g"]

    # Grafico
    fig, ax = plt.subplots()
    ax.plot(x, y, "bo", label="Dati sperimentali", marker="o", markersize=2, linestyle="none")
    ax.plot(x, model(x, a_fit, b_fit, g_fit), "r-", label="Fit")
    ax.set_xlabel("Lk")
    ax.set_ylabel("Frequenza (GHz)")
    ax.legend()
    plt.title("Fit della frequenza minima in funzione di Lk")
    plt.show()

    return a_fit, b_fit, g_fit

def calcolo_lk (a, b, g, fr) :
    """
    Calcola Lk in funzione di fr
    """
    Lk = (a/(fr-b))**2 - g
    return Lk
    
    