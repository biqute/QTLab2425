import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from matplotlib.gridspec import GridSpec
import scipy.optimize as spopt
from scipy import stats
from scipy.special import iv, kv
from scipy.constants import hbar, k, h 


"""
def model(T, Q0, Delta0_meV, fr, alpha=0.8):
    meV_to_J = 1.60218e-22  # 1 meV in Joules
    Delta0 = Delta0_meV * meV_to_J
    T = np.asarray(T, dtype=float)
    T = T * 10e-3  #conversione mK->K
    # Correzione: icisi e uso di Bessel su valori scalari
    xi = (h*fr)/(2*k*T) # Calcola una lista/array dei valori
    bessel_term = np.sinh(xi) * kv(0, xi)  # Funzione vettorializzata
    iv_term = np.exp(-xi) * iv(0, -xi)
    num = np.exp(-Delta0/(k*T)) * bessel_term
    den = 1 - 2*np.exp(-Delta0/(k*T)) * iv_term
    
    return (1/Q0 + (2*alpha/np.pi)*(num/den))


def fit (Qi, T, freq, alpha_fixed=0.8):
    T = np.asarray(T, dtype=float)
    Qi_err = np.ones(len(Qi))*10
    least_squares = LeastSquares(T, Qi, Qi_err, model)
    
    minuit = Minuit (least_squares,
                        Q0 = max(Qi),
                        Delta0_meV = 0.1,
                        fr = freq,
                        alpha = alpha_fixed)
    
    minuit.limits["Q0"] = (None, None)
    minuit.limits["Delta0_meV"] = (None, None)
    minuit.fixed["fr"] = True
    minuit.fixed["alpha"] = True

    minuit.migrad ()  # finds minimum of least_squares function
    minuit.hesse ()   # accurately computes uncertainties

    # global characteristics of the fit
    is_valid = minuit.valid
    print ('success of the fit: ', is_valid)

    Chi_quadro = minuit.fval/minuit.ndof
    print ('Chi quadro ridotto: ', Chi_quadro)

    for par, val, err in zip (minuit.parameters, minuit.values, minuit.errors) :
        print(f'{par} = {val:.3f} +/- {err:.3f}') # formatted output
    
    Q0_fit = minuit.values[0]
    Delta0_fit = minuit.values[1]

    fig, ax = plt.subplots ()
    plt.plot (T, 1/np.array(Qi))
    plt.plot (T, model(Q0_fit, Delta0_fit, T, freq))
    plt.title ("fit di risonanze")
    plt.show()


"""
def model(T, fr, Q0, Delta0_meV, alpha=0.8):
    meV_to_J = 1.60218e-22  # 1 meV in Joules
    Delta0 = Delta0_meV * meV_to_J
    T = np.asarray(T, dtype=float) * 1e-3  # conversione mK -> K
    
    xi = (h * fr) / (2 * k * T)  
    bessel_term = np.sinh(xi) * kv(0, xi)  
    iv_term = np.exp(-xi) * iv(0, -xi)

    num = np.exp(-Delta0 / (k * T)) * bessel_term
    den = 1 - 2 * np.exp(-Delta0 / (k * T)) * iv_term
    
    return 1/Q0 + (2 * alpha / np.pi) * (num / den)


# Funzione di fit
def fit(Qi, T, freq, alpha_fixed=0.8):
    T = np.asarray(T, dtype=float)  
    Qi_err = np.ones(len(Qi)) * 10 
    inv_Qi_err = np.array(Qi_err) /np.array(Qi)**2
    
    least_squares = LeastSquares(T, 1/np.array(Qi), inv_Qi_err, model)
    
    minuit = Minuit(least_squares,
                    fr=freq,
                    Q0=max(Qi),
                    Delta0_meV=0.1,
                    alpha=alpha_fixed)

    # Imposta limiti e fissa parametri
    minuit.limits["Q0"] = (None, None)
    minuit.limits["Delta0_meV"] = (None, None)
    minuit.fixed["fr"] = True
    minuit.fixed["alpha"] = True

    # Esegui il fit
    minuit.migrad()
    minuit.hesse()

    # Stampa risultati
    print("Successo del fit:", minuit.valid)
    print("Chi-quadro ridotto:", minuit.fval / minuit.ndof)
    for par, val, err in zip(minuit.parameters, minuit.values, minuit.errors):
        print(f"{par} = {val:.3f} ± {err:.3f}")

    # Estrai parametri
    Q0_fit = minuit.values["Q0"]
    Delta0_fit = minuit.values["Delta0_meV"]

    # Grafico
    fig, ax = plt.subplots()
    ax.plot(T, 1/np.array(Qi), "bo", label="Dati sperimentali", marker="o", markersize=2, linestyle="none")
    ax.plot(T, model(T, freq, Q0_fit, Delta0_fit), "r-", label="Fit")
    ax.set_xlabel("Temperatura (mK)")
    ax.set_ylabel("1/Qi")
    ax.legend()
    plt.title("Fit della qualità interna")
    plt.show()
