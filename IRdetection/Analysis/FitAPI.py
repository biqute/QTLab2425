from iminuit import Minuit
from iminuit.cost import LeastSquares
import numpy as np
from typing import Callable, Optional

class Model:
    def __init__(self, model_function: Callable, param_names: list[str]):
        self.model_function = model_function
        self.param_names = param_names
        for name in param_names:
            setattr(self, name, None)
    
    def assing_params(self, params: dict[str, float]) -> None:
        for p_name in self.param_names:
            setattr(self, p_name, params[p_name])
    
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
                
        
    def __call__(self, data, **kwargs) -> np.ndarray:
        # create the active model function. i.e. model function with only the active parameters
        param_dict = {}
        param_iter = iter(kwargs.values())
        # first params is the data array
        for p_name in self.param_names:
            if self.active_params.get(p_name): # if the parameter is active
                param_dict[p_name] = next(param_iter)
            else: # if the parameter is not active
                param_dict[p_name] = getattr(self, p_name)
        return self.model_function(data, **param_dict)

    
    
class Fitter:
    def __init__(self, 
                 model_function: Callable, 
                 param_names: list[str], 
                 data: np.ndarray,
                 loss_function: Callable,
                 yerr: Optional[np.ndarray] = None, 
                 xerr: Optional[np.ndarray] = None, 
                 params_initial_guess: Optional[dict[str, float]] = None, 
                 params_range: Optional[dict[str, tuple[float, float]]] = None):
        # Check if the data is valid
        if  0 < len(data[0]) < 3:
            raise ValueError("Data must be 1D or 2D")
        if len(data[0]) == 1:
            raise NotImplementedError("1D data is not supported yet")
        
        self.x = data[:, 0]
        self.y = data[:, 1]
        self.yerr = yerr
        self.xerr = xerr
        # Define the model
        self.model = Model(model_function, param_names)
        # Assign the initial guess to the model if provided
        if params_initial_guess is not None:
            self.model.assing_params(params_initial_guess)
        
        self.loss_function = loss_function  
        self.params_range = params_range
    
    def fit_quasi_magicus(self, N_fixed_params: Optional[int] = None):
        """
        Fit the data using quasi-magicus method
        
        NOTE: It's Leviosa not Leviosar
        """
        if N_fixed_params is None:
            N_fixed_params = len(self.model.param_names)//2 #default fixed params is half of the total params
        
        # Set the fixed params
        self.model.set_fixed_params(self.model.param_names[:N_fixed_params])
        
        #partial fit
        
        
        
        
    def fit(self):
        if self.xerr is None:
            loss = self.loss_function(self.x, self.y, self.yerr, self.model)
        else:
            print("WARNING: You are using x errors. Please make sure your loss function supports x errors")
            loss = self.loss_function(self.x, self.y, self.xerr, self.yerr, self.model)
        
        active_params_names = [p_name for p_name, is_active in self.model.active_params.items() if is_active]
        active_params = {p_name: getattr(self.model, p_name) for p_name in active_params_names}
        m = Minuit(loss, **active_params)

        #set params limits
        if self.params_range is not None:
            for p_name, p_range in self.params_range.items():
                m.limits[p_name] = p_range
                
        m.migrad()
        return m


def model_func(x: float, a: float, b: float) -> float:
    return a * x + b

model = Model(model_func, ["a", "b"])
model.assing_params({"a": 1, "b": 2})
model.set_active_params(["a"])
data = np.array([1, 2, 3, 4, 5])
res = model(data, a=3)
print(res)