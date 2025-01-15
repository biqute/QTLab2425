import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from matplotlib.gridspec import GridSpec
import scipy.optimize as spopt
from scipy import stats
import re


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
    amp = - amp / np.min(amp)     
    return f, amp


def converter (amp, unit="linear"):
    '''
    Funzione per convertire dBm to mW o viceversa
    '''
    if unit == "linear":
        return 10**(amp/20)  
    else: return 20 * np.log10(amp)


def extract_temperature(filename):
    """
    Estrae la temperatura da un nome file, supportando mK e K.
    """
    match = re.search(r'(\d+(?:\.\d+)?)(mK|K)', filename)
    if match:
        temp = float(match.group(1))
        unit = match.group(2)
        # Converti in millikelvin se necessario
        return temp if unit == "mK" else temp * 1000
    else:
        raise ValueError(f"Temperatura non trovata nel nome del file: {filename}")
    


def process_files(file_list):
    """
    Itera sui file e restituisce un array di temperature.
    
    Parametri:
    - file_list: Lista di nomi file (es. ["200mK_-20dBm.txt", ...])
    
    Ritorna:
    - Lista di temperature in millikelvin.
    """
    temperatures = []
    for filename in file_list:
        try:
            temperature = extract_temperature(filename)
            temperatures.append(temperature)
        except ValueError as e:
            print(e)
    return temperatures


def model (f, a, b, c, d, k, phi, Qt, Qc, fr):
    x = f - fr 
    return a + b*x + c*x**2 + d*x**3 + k*np.abs(1 - Qt*np.exp(1j*phi) / np.abs(Qc) / (1 + 2j*Qt*x/fr))


def fit(file_path, unit = "linear"):
    f, amp = read_data(file_path)
    amp_err = np.ones(len(amp))*10e-5
    amp = converter(amp, unit)

    i = np.argmin(amp)
    fmin = f[i]
    amp_min = min (amp)
    amp_max = max (amp)
    amp_FWHM = amp_min + (amp_max - amp_min)/2

    # Trova gli indici dove y è vicino a y_half
    tolleranza = 0.2*1e-3  # Puoi regolare la tolleranza a seconda dei tuoi dati
    indici = np.where(np.abs(amp - amp_FWHM) < tolleranza)[0]
    if (len(indici) < 2):
        print ("non è stato possibile trovare due indici, aumentare la tolleranza")

    # Trova le frequenze corrispondenti
    frequenze_half = f[indici]

    Qc_guess = 1000
    Qt_guess = fmin / np.abs(frequenze_half[0] - frequenze_half[-1])

    print ("frequenza di FWHM sx: ", frequenze_half[0],"frequenza di FWHM dx: ", frequenze_half[-1], "Qt guessed: ", Qt_guess)

    k_guess = ( np.max(amp) - np.min(amp) ) * (Qc_guess/Qt_guess)


    least_squares = LeastSquares(f, amp, amp_err, model)

    minuit = Minuit (least_squares,
                        a = 1,
                        b = 10e-9,
                        c = 10e-18,
                        d = 10e-27,
                        k = k_guess,
                        phi = 0.01,
                        Qt = Qt_guess,
                        Qc = Qc_guess,
                        fr = fmin)
    
    minuit.limits["a"] = (None, None)
    minuit.limits["b"] = (None, None)
    minuit.limits["c"] = (None, None)
    minuit.limits["d"] = (None, None)
    minuit.limits["k"] = (0, None)
    minuit.limits["phi"] = (-np.pi, np.pi)
    minuit.limits["Qt"] = (0, 1000000)
    minuit.limits["Qc"] = (0, 1000000)
    minuit.limits["fr"] = (fmin - 1e6, fmin + 1e6)

    minuit.migrad ()  # finds minimum of least_squares function
    minuit.hesse ()   # accurately computes uncertainties

    # global characteristics of the fit
    is_valid = minuit.valid
    print ('success of the fit: ', is_valid)

    Chi_quadro = minuit.fval/minuit.ndof
    print ('Chi quadro ridotto: ', Chi_quadro)

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
    plt.plot (f, model(f, a_fit, b_fit, c_fit, d_fit, k_fit, phi_fit, Qt_fit, Qc_fit, fr_fit))
    plt.axhline(y=amp_FWHM, color='red', linestyle='--')
    plt.title ("fit di risonanze")
    plt.show()

    return fr_fit, Qc_fit



def main () :

    file_path = "20mK_-20dBm.txt"
    fit(file_path)

if __name__ == "__main__":
    main()