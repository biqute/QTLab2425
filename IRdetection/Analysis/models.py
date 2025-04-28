# Example usage
import numpy as np
from scipy import special as sp
from scipy import constants as cs

def resonance_model(f: np.ndarray, f0: float, phi: float, Qt: float, Qc: float, A: float, B: float, C: float, D: float, K: float, fmin: float) -> np.ndarray:
    return (A+B*(f-fmin) + C*(f-fmin)**2 + D*(f-fmin)**3) + K * np.abs((1 - (Qt/np.abs(Qc))*np.exp(1j*phi)/(1 + 2j*Qt*((f-fmin) - f0)/fmin)))

def parametric_resonator_peak_vs_bias_current(i: np.ndarray, a: float, b: float) -> np.ndarray: # F(I) = I^2/a + I^4/b (+c)
    i = np.array(i.copy())  # Ensure i is a numpy array
    return -0.5 * ((i**2 / a**2) + (i**4 / b**4))  # + c

def linear(x: np.ndarray, a: float, b: float) -> np.ndarray:
    x = np.array(x.copy())  # Ensure x is a numpy array
    return a * x + b

def quadratic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    x = np.array(x.copy())  # Ensure x is a numpy array
    return a * x**2 + b * x + c    

def qi_factor_model(T: np.ndarray, a: float, w: float, Q0: float, D0_k: float) -> np.ndarray:
    T=np.array(T.copy())/1000
    csi = cs.hbar * w / (2 * cs.k * T)
    exp = np.exp(-D0_k/T)
    return Q0**-1 + (2*a / np.pi) * exp* np.sinh(csi) * sp.kv(0, csi) / (1 - 2 * exp*np.exp(-csi)*sp.iv(0, -csi))