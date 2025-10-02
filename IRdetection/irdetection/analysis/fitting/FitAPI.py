from iminuit import Minuit
from iminuit.cost import LeastSquares
import numpy as np
from scipy import stats
from typing import Callable, Optional
from irdetection.analysis.fitting.searcher import Searcher
from irdetection.analysis.fitting.models import Model

    
class Fitter:
    """
    Fitter class for fitting data using a model and a loss function.
    This class provides a unified interface for fitting data with various models and loss functions.
    It allows for flexible parameter management, including initial guesses, parameter ranges, and error handling.
    It also supports the use of searchers to find good initial guesses for the fit.
    The fitting process is done using the Minuit library.
    
    Parameters
    ----------
    model : Model
        The model to fit the data with. It should be an instance of the Model class.
    param_names : list[str]
        List of parameter names to be used in the model.
    data : np.ndarray
        The data to fit, should be a 2D array with shape (N, 2) where N is the number of samples.
    loss_manager : Callable, optional
        The loss manager to use for the fitting process. It must be a *Class* and shuld be compatible with Minuit.
        Default is LeastSquares. (Note if you want to use xerr, you should use a custom loss manager, LeastSquares does not support xerr and will raise an error)
    loss_function : Callable | str, optional
        The loss function to use for the fitting process. It can be a string or a callable function.
        If a string is provided, it should be one of the predefined loss functions in the loss module.
        Default is "linear" which is the default of LeastSquares.
    yerr : np.ndarray, optional
        The error in the y values. If None, it will be estimated as 5% of the data if greater than 1e-5 else 1e-5.
        Default is None.
    xerr : np.ndarray, optional
        The error in the x values. If None, it will be set to None.
        Default is None.
    params_initial_guess : dict[str, float], optional
        Initial guess for the parameters of the model. If None, the model will use its default initial guess.
        Default is None.
    params_range : dict[str, tuple[float, float]], optional
        Range for the parameters of the model. If None, the model will use its default range.
        Default is None.
    
    Raises
    ------
    ValueError
        If the data is not a 2D array with shape (N, 2) or if it has less than two samples.
        
    """
    def __init__(self, 
                 model_function: Callable, 
                 param_names: list[str], 
                 data: np.ndarray,
                 loss_manager = LeastSquares,
                 loss_function: Callable | str = "linear",
                 yerr: Optional[np.ndarray] = None, 
                 xerr: Optional[np.ndarray] = None, 
                 params_initial_guess: Optional[dict[str, float]] = None, 
                 params_range: Optional[dict[str, tuple[float, float]]] = None):
        
        # Check if the data is valid
        if len(data.shape) != 2 or data.shape[1] != 2:
            raise ValueError("Data must be a 2D array with shape (N, 2) where N is the number of samples")
        if data.shape[0] < 2:
            raise ValueError("Data must have at least two samples")
        
        self.x = data[:, 0]
        self.y = data[:, 1]
        self.data = data  # Store the original data for use in fit_magicus
        self.yerr = yerr
        self.xerr = xerr
        # Define the model
        self.model = Model(model_function, param_names)
        # Assign the initial guess to the model if provided
        if params_initial_guess is not None:
            self.model.assing_params(params_initial_guess)
        
        self.loss_manager = loss_manager
        self.loss_function = loss_function  
        self.params_range = params_range

    
        
    def fit(self, searcher = None) -> Minuit:
        """
        Fit the data using the model and the loss function.

        Parameters
        ----------
        searcher : Searcher, optional
            Searcher object to find a good initial guess for the fit. If None, uses the initial guess provided in the constructor.

        Returns
        -------
        Minuit
            Minuit object containing the fit result.
        """
        if searcher is not None:
            # Check if the searcher is valid
            if not isinstance(searcher, Searcher):
                raise ValueError("Searcher must be an instance of Searcher class")
            # Search for the best initial guess
            searcher.search(self.data, self.model)
            # Fit the model with the found parameters
            self.model.assing_params(searcher.params)
            #set params boundaries
            

        active_params_names = [p_name for p_name, is_active in self.model.active_params.items() if is_active]
        active_params = {p_name: getattr(self.model, p_name) for p_name in active_params_names}

        if self.yerr is None: # Estimate the error as 5% of the data if greater than 1e-5 else 1e-5: bigger error is needed to make every fit converge!
            self.yerr = np.maximum(1e-5, 1 * self.y)
        if self.xerr is None:
            loss = self.loss_manager(x=self.x, y=self.y, yerror=self.yerr, model=self.model, loss=self.loss_function, name=active_params_names)
            
        else:
            loss = self.loss_manager(x=self.x, y=self.y, xerror=self.xerr, yerror=self.yerr, model=self.model, loss=self.loss_function, name=active_params_names)
        
        self.m = Minuit(loss, **active_params)

        #set params limits
        for p_name, p_range in self.model.param_bounds.items():
            if p_name in self.m.parameters:
                self.m.limits[p_name] = p_range
            
        self.m.migrad()
        return self.m
    

    def p_value(self) -> float:
        return 1 - stats.chi2.cdf(self.m.fval, self.m.ndof)
    

# GRAVEYARD: Here lies old code, made with love and care, never worked, but still loved.

# def fit_quasi_magicus(self, N_fixed_params: Optional[int] = None, pvalue_treshold: Optional[float] = 0.005, try_total_fit_first: Optional[bool] = True):
#         """
#         Fit the data using quasi-magicus method
        
#         NOTE: It's Leviosa not Leviosar
#         """
#         # Fit the model with all the parameters 
#         if try_total_fit_first:
#             self.model.set_active_params(self.model.param_names)
#             fit_result = self.fit()
#             if fit_result.valid:
#                 if self.p_value(fit_result) > pvalue_treshold:
#                     print(f"First total fit converged good, p-value = {self.p_value(fit_result)}")
#                     return fit_result
#                 else:
#                     print(f"First total fit converged bad, p-value = {self.p_value(fit_result)}")
#             else:
#                 print("First total fit failed")
        
#         if N_fixed_params is None:
#             N_fixed_params = len(self.model.param_names)//2 #default fixed params is half of the total params
        
#         params_to_fit = self.model.param_names
#         #partial fit
#         iter_counter = 0
#         while len(params_to_fit) > 0:
#             print(f"Partial fit {iter_counter}")
#             # Set the fixed params
#             N_fixed_params = min(N_fixed_params, len(params_to_fit))
#             self.model.set_fixed_params(self.model.param_names[:N_fixed_params])
#             # Fit the model with the fixed params
#             fit_result = self.fit()
#             # Check if the fit converged
#             if fit_result.valid:
#                 # Check if the p-value is greater than the treshold
#                 if self.p_value(fit_result) > pvalue_treshold: # Fit is good
#                     # pop first N_fixed_params
#                     params_to_fit = params_to_fit[N_fixed_params:]
#                     # Set inital guess of currently active params to the fitted values
#                     result_values = fit_result.values.to_dict()
#                     self.model.assing_params({p_name: result_values[p_name] for p_name in result_values})
#                     print(f"Partial fit {iter_counter} converged good , p-value = {self.p_value(fit_result)}")
#                 else: # Fit is bad
#                     # put first element at the end
#                     params_to_fit = params_to_fit[1:] + [params_to_fit[0]]  
#                     print(f"Partial fit {iter_counter} converged bad , p-value = {self.p_value(fit_result)}")          
#             else:
#                 # put first element at the end
#                 params_to_fit = params_to_fit[1:] + [params_to_fit[0]]
#                 print(f"Partial fit {iter_counter} did not converge")
            
#             # Check if all partial fits failed
#             if iter_counter == len(self.model.param_names):
#                 raise RuntimeError("The fit did not converge. All partial fits failed")
#             iter_counter += 1
        
#         # Final fit with all params
#         self.model.set_active_params(self.model.param_names)
#         return self.fit()