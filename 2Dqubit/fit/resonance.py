import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from matplotlib.gridspec import GridSpec
import scipy.optimize as spopt
from scipy import stats


def read_data(file_path):
    '''
    funzione per leggere i dati
    '''
    data = np.loadtxt(file_path, delimiter = None)
    num_columns = data.shape[1]
    f = data[:, 0]    
    if num_columns == 3:  
        I = data[:, 1]     
        Q = data[:, 2]
        amp = np.sqrt(I**2 + Q**2)
    else: amp = data[:, 1]
    amp /= np.min(amp)     
    return f, amp


def converter (amp, unit="linear"):
    '''
    Funzione per convertire dBm to mW o viceversa
    '''
    if unit == "linear":
        return 10**(amp/20)  
    else: return 20 * np.log10(amp)


def model (f, a, b, c, d, k, phi, Qt, Qc, fr):
    x = f - fr 
    return a + b*x + c*x**2 + d*x**3 + k*np.abs(1 - Qt*np.exp(1j*phi) / np.abs(Qc) / (1 + 2j*Qt*x/fr))


def fit(file_path, unit = "linear"):
    f, amp = read_data(file_path)
    amp_err = np.ones(len(amp))*10e-5
    amp = converter(amp, unit)

    i = np.argmin(amp)
    fmin = f[i]
        
    least_squares = LeastSquares(f, amp, amp_err, model)

    minuit = Minuit (least_squares,
                        a = 0,
                        b = 0,
                        c = 0,
                        d = 0,
                        k = 0,
                        phi = 0,
                        Qt = 0,
                        Qc = 0,
                        fr = fmin)
    
    minuit.limits["a"] = (None, None)
    minuit.limits["b"] = (None, None)
    minuit.limits["c"] = (None, None)
    minuit.limits["d"] = (None, None)
    minuit.limits["k"] = (0, None)
    minuit.limits["phi"] = (-np.pi, np.pi)
    minuit.limits["Qt"] = (0, None)
    minuit.limits["Qc"] = (0, None)
    minuit.limits["fr"] = (0, None)

    minuit.migrad ()  # finds minimum of least_squares function
    minuit.hesse ()   # accurately computes uncertainties

    # global characteristics of the fit
    is_valid = minuit.valid
    print ('success of the fit: ', is_valid)

    for par, val, err in zip (minuit.parameters, minuit.values, minuit.errors) :
        print(f'{par} = {val:.3f} +/- {err:.3f}') # formatted output
    
    a_fit = minuit.values[0]
    b_fit = minuit.values[1]
    c_fit = minuit.values[2]
    d_fit = minuit.values[3]
    k_fit = minuit.values[4]
    phi_fit = minuit.values[5]
    Qt_fit = minuit.values[6]
    Qc_fit = minuit.values[7]
    fr_fit = minuit.values[8]

    fig, ax = plt.subplots ()
    plt.plot (f, amp)
    plt.plot (f, model(a_fit, b_fit, c_fit, d_fit, k_fit, phi_fit, Qt_fit, Qc_fit, fr_fit))
    plt.show()





 
