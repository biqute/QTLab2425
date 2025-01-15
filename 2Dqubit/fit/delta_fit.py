import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from matplotlib.gridspec import GridSpec
import scipy.optimize as spopt
from scipy import stats
from scipy.special import iv, kv
from scipy.constants import hbar, k, h 



def model(T, Q0, Delta0, fr, alpha=0.8):

    T = np.array(T)

    # Correzione: icisi e uso di Bessel su valori scalari
    icsi_values = np.array((h*fr)/(2*k*T)) # Calcola una lista/array dei valori
    bessel_term = (np.sinh(icsi_values) * kv(0, icsi_values))  # Funzione vettorializzata
    iv_values = np.array(iv(0, -icsi_values))
    iv_term = np.exp(-(Delta0/(k*T))) * np.exp(-icsi_values) * iv_values
    iv_term = np.array (iv_term)
    
    return (1/Q0 + alpha*((4/np.pi)*(np.exp(-(Delta0/(k*T))) * bessel_term)/(1 - 2 * iv_term)))


def icsi (fr, T):
    T = np.array(T)
    return (h*fr)/(2*k*T)


def fit (Qc, T, freq, alpha_fixed=0.8):

    Qc_err = np.ones(len(Qc))*10
    least_squares = LeastSquares(T, Qc, Qc_err, model)
    
    minuit = Minuit (least_squares,
                        Q0 = max(Qc),
                        Delta0 = 1,
                        fr = freq,
                        alpha = alpha_fixed)
    
    minuit.limits["Q0"] = (None, None)
    minuit.limits["Delta0"] = (None, None)
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
    plt.plot (T, 1/np.array(Qc))
    plt.plot (T, model(Q0_fit, Delta0_fit, T, freq))
    plt.title ("fit di risonanze")
    plt.show()