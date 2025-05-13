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
