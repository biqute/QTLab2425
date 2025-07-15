from iminuit import Minuit
from iminuit.cost import LeastSquares
import numpy as np
from scipy import stats
from typing import Callable, Optional

class Model_old:
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
                
        
    def __call__(self, data, *args) -> np.ndarray:
        # create the active model function. i.e. model function with only the active parameters
        param_dict = {}
        param_iter = iter(args)
        for p_name in self.param_names:
            if self.active_params.get(p_name):  # if the parameter is active
                param_dict[p_name] = next(param_iter)
            else:  # if the parameter is not active
                param_dict[p_name] = getattr(self, p_name)
        return self.model_function(data, **param_dict)

    
    
class Fitter_old:
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
        if not 0 < len(data[0]) < 3:
            raise ValueError("Data must be 1D or 2D")
        if len(data[0]) == 1:
            raise NotImplementedError("1D data is not supported yet")
        
        self.x = data[:, 0]
        self.y = data[:, 1]
        self.yerr = yerr
        self.xerr = xerr
        # Define the model
        self.model = Model_old(model_function, param_names)
        # Assign the initial guess to the model if provided
        if params_initial_guess is not None:
            self.model.assing_params(params_initial_guess)
        
        self.loss_function = loss_function  
        self.params_range = params_range
    
    def p_value(self, m: Minuit) -> float:
        return 1 - stats.chi2.cdf(m.fval, m.ndof)
    
    def fit_quasi_magicus(self, N_fixed_params: Optional[int] = None, pvalue_treshold: Optional[float] = 0.005, try_total_fit_first: Optional[bool] = True):
        """
        Fit the data using quasi-magicus method
        
        NOTE: It's Leviosa not Leviosar
        """
        # Fit the model with all the parameters 
        if try_total_fit_first:
            self.model.set_active_params(self.model.param_names)
            fit_result = self.fit()
            if fit_result.valid:
                if self.p_value(fit_result) > pvalue_treshold:
                    print(f"First total fit converged good, p-value = {self.p_value(fit_result)}")
                    return fit_result
                else:
                    print(f"First total fit converged bad, p-value = {self.p_value(fit_result)}")
            else:
                print("First total fit failed")
        
        if N_fixed_params is None:
            N_fixed_params = len(self.model.param_names)//2 #default fixed params is half of the total params
        
        params_to_fit = self.model.param_names
        #partial fit
        iter_counter = 0
        while len(params_to_fit) > 0:
            print(f"Partial fit {iter_counter}")
            # Set the fixed params
            N_fixed_params = min(N_fixed_params, len(params_to_fit))
            self.model.set_fixed_params(self.model.param_names[:N_fixed_params])
            # Fit the model with the fixed params
            fit_result = self.fit()
            # Check if the fit converged
            if fit_result.valid:
                # Check if the p-value is greater than the treshold
                if self.p_value(fit_result) > pvalue_treshold: # Fit is good
                    # pop first N_fixed_params
                    params_to_fit = params_to_fit[N_fixed_params:]
                    # Set inital guess of currently active params to the fitted values
                    result_values = fit_result.values.to_dict()
                    self.model.assing_params({p_name: result_values[p_name] for p_name in result_values})
                    print(f"Partial fit {iter_counter} converged good , p-value = {self.p_value(fit_result)}")
                else: # Fit is bad
                    # put first element at the end
                    params_to_fit = params_to_fit[1:] + [params_to_fit[0]]  
                    print(f"Partial fit {iter_counter} converged bad , p-value = {self.p_value(fit_result)}")          
            else:
                # put first element at the end
                params_to_fit = params_to_fit[1:] + [params_to_fit[0]]
                print(f"Partial fit {iter_counter} did not converge")
            
            # Check if all partial fits failed
            if iter_counter == len(self.model.param_names):
                raise RuntimeError("The fit did not converge. All partial fits failed")
            iter_counter += 1
        
        # Final fit with all params
        self.model.set_active_params(self.model.param_names)
        return self.fit()
    
    def fit_non_magicus(self):
        self.model.set_active_params(self.model.param_names)
        return self.fit()
        
    def fit(self):
        active_params_names = [p_name for p_name, is_active in self.model.active_params.items() if is_active]
        active_params = {p_name: getattr(self.model, p_name) for p_name in active_params_names}

        if self.yerr is None: # Estimate the error as 1% of the data if greater than 1e-5 else 1e-5
            self.yerr = np.maximum(1e-5, 0.01 * self.y)
        if self.xerr is None:
            loss = self.loss_function(self.x, self.y, self.yerr, self.model, name=active_params_names)
        else:
            loss = self.loss_function(self.x, self.y, self.xerr, self.yerr, self.model, name=active_params_names)
        
        m = Minuit(loss, **active_params)

        #set params limits
        if self.params_range is not None:
            for p_name, p_range in self.params_range.items():
                m.limits[p_name] = p_range

                
        m.migrad()
        return m



