import numpy as np
from scipy import special as sp
from scipy import constants as cs
from typing import Callable
#import torch


class Model:
    def __init__(self, model_function: Callable, param_names: list[str]):
        self.model_function = model_function
        self.param_names = param_names
        for name in param_names:
            setattr(self, name, None)
        #set another attribute to store the bounds for the parameters
        self.param_bounds = {p_name: (-np.inf, np.inf) for p_name in param_names}
        
        # By default all parameters are active and their value set to zero
        self.active_params = {}
        for p_name in param_names:
            self.active_params[p_name] = True
            setattr(self, p_name, 0.0)  # Set default value to 0.0
    
    def assing_params(self, params: dict[str, float]) -> None:
        for p_name in self.param_names:
            setattr(self, p_name, params[p_name])
            
    # define a method to assign the bounds for the parameters
    def set_param_bounds(self, bounds: dict[str, tuple[float, float]]) -> None:
        for p_name in bounds.keys():
            if p_name not in self.param_names:
                raise ValueError(f"{p_name} is not a valid parameter name")
            if not isinstance(bounds[p_name], tuple) or len(bounds[p_name]) != 2:
                raise ValueError(f"Bounds for {p_name} must be a tuple of two values (min, max)")
            self.param_bounds[p_name] = bounds[p_name]
    
    def set_active_params(self, active_params: list[str]) -> None:
        self.active_params = {}
        # check if the active params are valid
        for p_name in active_params:
            if p_name not in self.param_names:
                raise ValueError(f"{p_name} is not a valid parameter name")
        
        for p_name in self.param_names:
            if p_name in active_params:
                self.active_params[p_name] = True
            else:
                self.active_params[p_name] = False
                
    def set_fixed_params(self, fixed_params: list[str]) -> None:
        self.active_params = {}
        # check if the active params are valid
        for p_name in fixed_params:
            if p_name not in self.param_names:
                raise ValueError(f"{p_name} is not a valid parameter name")
            
        for p_name in self.param_names:
            if p_name in fixed_params:
                self.active_params[p_name] = False
            else:
                self.active_params[p_name] = True
                
    def __call__(self, data, *args) -> np.ndarray:
        # create the active model function. i.e. model function with only the active parameters
        param_dict = {}
        param_iter = iter(args)
        for p_name in self.param_names:
            if self.active_params.get(p_name):  # if the parameter is active
                param_dict[p_name] = next(param_iter)
            else:  # if the parameter is not active
                param_dict[p_name] = getattr(self, p_name)

        try:  
            result = self.model_function(data, **param_dict)
        except Exception as e:
            raise RuntimeError(f"Model function evaluation failed with error: {e}")
    
        return result
    
# MODEL FUNCTIONS ----------------------------------------------------    

def resonance_model(f: np.ndarray, f0: float, phi: float, Qt: float, Qc: float, A: float, B: float, C: float, D: float, K: float, fmin: float) -> np.ndarray:
    return (A+B*(f-fmin) + C*(f-fmin)**2 + D*(f-fmin)**3) + K * np.abs((1 - (Qt/np.abs(Qc))*np.exp(1j*phi)/(1 + 2j*Qt*((f-fmin) - f0)/fmin)))

def resonance_model_test(f: np.ndarray, f0: float, phi: float, Qt: float, Qc: float, A: float, B: float, C: float, D: float, K: float, fmin: float) -> np.ndarray:
    if Qt > Qc:
        return np.zeros_like(f)
    
    return (A+B*(f-fmin) + C*(f-fmin)**2 + D*(f-fmin)**3) + K * np.abs((1 - (Qt/np.abs(Qc))*np.exp(1j*phi)/(1 + 2j*Qt*((f-fmin) - f0)/fmin)))

def resonance_model_smart(f: np.ndarray, f0: float, phi: float, Qs: float, Qi: float, A: float, B: float, C: float, D: float, K: float, fmin: float) -> np.ndarray:
    return (A+B*(f-fmin) + C*(f-fmin)**2 + D*(f-fmin)**3) + K * np.abs((1 - (Qs)*np.exp(1j*phi)/(1 + Qs + 2j*Qi*((f-fmin) - f0)/fmin)))

def parametric_resonator_peak_vs_bias_current(i: np.ndarray, a: float, b: float) -> np.ndarray: # F(I) = I^2/a^2 + I^4/b^4\ (+c)
    i = np.array(i.copy())  # Ensure i is a numpy array
    return 0.5*((i**2 / a**2) + (i**4 / b**4))  # + c

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



# TORCH MODELS (for TorchFitter) ----------------------------------------------------
# def resonance_model_torch(
#     f: torch.Tensor,
#     f0: torch.Tensor, phi: torch.Tensor,
#     Qt: torch.Tensor, Qc: torch.Tensor,
#     A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor,
#     K: torch.Tensor, fmin: torch.Tensor
# ) -> torch.Tensor:
#     poly = A + B*(f - fmin) + C*(f - fmin)**2 + D*(f - fmin)**3
#     lorentz = 1 - (Qt / torch.abs(Qc)) * torch.exp(1j * phi) / (
#         1 + 2j * Qt * ((f - fmin) - f0) / fmin
#     )
#     return poly + K * torch.abs(lorentz)