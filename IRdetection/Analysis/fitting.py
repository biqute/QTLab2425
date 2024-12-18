from iminuit import Minuit
import numpy as np
from typing import Callable, Optional

def chi2(y: np.ndarray, yerr: Optional[np.ndarray], model: Callable, params: np.ndarray) -> float:
    if yerr is None:
        yerr = np.ones_like(y)
    return np.sum(((y - model(*params)) / yerr) ** 2)

def S21_model(f: np.ndarray, f_min: float, f0: float, phi: float, Qt: float, Qc: float, A: float, B: float, C: float, D: float, K: float) -> np.ndarray:
    complex = (A + B*f + C*f**2 + D*f**3) + K * (1 - (Qt/Qc)*np.exp(1j*phi)/(1 - 2j*Qt*(f - f0)/f_min))
    return np.abs(complex)

def fit_S21(f: np.ndarray, S21: np.ndarray, S21_err: Optional[np.ndarray] = None, f0_guess: Optional[float] = None, phi_guess: Optional[float] = None, Qt_guess: Optional[float] = None, Qc_guess: Optional[float] = None, A_guess: Optional[float] = None, B_guess: Optional[float] = None, C_guess: Optional[float] = None, D_guess: Optional[float] = None, K_guess: Optional[float] = None) -> Minuit:
    f = f.copy()
    # Shift the frequency axis to zero
    f -= f[np.argmin(S21)]
    # Normalize the S21 data
    S21 /= np.max(S21)

    if f0_guess is None:
        f0_guess = 0.01
    if phi_guess is None:
        phi_guess = 0.01
    if Qt_guess is None:
        Qt_guess = 1e3
    if Qc_guess is None:
        Qc_guess = 1e3
    if A_guess is None:
        A_guess = 1.0
    if B_guess is None:
        B_guess = 1e-9
    if C_guess is None:
        C_guess = 1e-18
    if D_guess is None:
        D_guess = 1e-27
    if K_guess is None:
        K_guess = 1.0
        
    # Define the fitting model
    def fit_model(f0, phi, Qt, Qc, A, B, C, D, K):
        f_min = 100
        return chi2(S21, S21_err, S21_model, (f, f_min, f0, phi, Qt, Qc, A, B, C, D, K))
        

    m = Minuit(fit_model, f0=f0_guess, phi=phi_guess, Qt=Qt_guess, Qc=Qc_guess, A=A_guess, B=B_guess, C=C_guess, D=D_guess, K=K_guess)
    m.migrad()
    return m



def plot_fit(m: Minuit, x: np.ndarray, y: np.ndarray, yerr: Optional[np.ndarray] = None, linspace: Optional[np.ndarray] = None, **kwargs) -> None:
    import matplotlib.pyplot as plt
    
    if linspace is None:
        linspace = x
    if yerr is None:
        plt.scatter(x, y, **kwargs, label='Data')
    else:
        plt.errorbar(x, y, yerr=yerr, **kwargs, label='Data')
    fit_params = get_fit_params_list(m)
    #add f_min to fit_params
    fit_params.insert(1, 100)
    plt.plot(linspace, S21_model(linspace, *fit_params), label='Fit', c='r')
    plt.legend()
    plt.show()
    
    
    
# ----------- UTILITY FUNCTIONS ------------

def get_fit_params_list(m: Minuit) -> list:
    return [m.values[key] for key in m.parameters]