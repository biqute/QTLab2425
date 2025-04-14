from iminuit import Minuit
from iminuit.cost import LeastSquares
import numpy as np
from scipy import stats
from typing import Callable, Optional
from abc import ABC, abstractmethod

class Model:
    def __init__(self, model_function: Callable, param_names: list[str]):
        self.model_function = model_function
        self.param_names = param_names
        for name in param_names:
            setattr(self, name, None)
        
        # By default all parameters are active and their value set to zero
        self.active_params = {}
        for p_name in param_names:
            self.active_params[p_name] = True
            setattr(self, p_name, 0.0)  # Set default value to 0.0
    
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
        
        self.loss_function = loss_function  
        self.params_range = params_range


    def fit_magicus(self, searcher):
        """
        It will fit the data.

        Uses a searcher to find good initial guess for the fit.
        """
        # Check if the searcher is valid
        if not isinstance(searcher, Searcher):
            raise ValueError("Searcher must be an instance of Searcher class")
        
        # Search for the best initial guess
        searcher.search(self.data, self.model)
        
        # Fit the model with the found parameters
        self.model.assing_params(searcher.params)
        return self.fit()
    
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

    def p_value(self, m: Minuit) -> float:
        return 1 - stats.chi2.cdf(m.fval, m.ndof)



class Searcher(ABC):
    def __init__(self):
        """
        Searcher is tasked with searching good initial guess for the fit.
        """
        self.params = {}

    @abstractmethod
    def search(self, data: np.ndarray, model: Model) -> None:
        """
        Search for the best initial guess for the fit.
        """
        pass
    
    def _validate_search(self) -> None:
        """
        Ensures that no parameter is None.
        """
        for p_name, p_value in self.params.items():
            if p_value is None:
                raise ValueError(f"Parameter {p_name} is not set")


class GridSearcher(Searcher):
    def __init__(self, center: float, stride: float, number_of_points: int):
        """
        Searcher that searches for the best initial guess using a grid search.

        Parameters
        :param center: Center of the grid search.
        :param stride: Stride of the grid search.
        :param number_of_points: Number of points in the grid search.
        """
        super().__init__()
        self.center = center
        self.stride = stride
        self.number_of_points = number_of_points
        self.grid_points = np.linspace(center - stride * number_of_points / 2, center + stride * number_of_points / 2, number_of_points)


class ResonancePeakSearcher(Searcher):
    def __init__(self):
        super().__init__()
        self.params = {
            "f0": None,
            "phi": None,
            "Qt": None,
            "Qc": None,
            "A": None,
            "B": None,
            "C": None,
            "D": None,
            "K": None,
            "fmin": None
        }

    def search(self, data: np.ndarray, model: Model) -> None:
        """
        Search for the best initial parameters for the fit.

        Parameters
        ----------
        data : np.ndarray
            2D numpy array with frequency and dBm amplitude values.
            Assumes data[:, 0] = frequency and data[:, 1] = dBm amplitude.
        model : Model
            Model object with the model function and parameter names.
        """
        # Verify that model parameters match
        for p_name in self.params.keys():
            if p_name not in model.param_names:
                raise ValueError(f"{p_name} is not a valid parameter name, params are {model.param_names}")
        
        # Check that data is a valid 2D numpy array with 2 columns and no NaNs
        if len(data.shape) != 2 or data.shape[1] != 2:
            raise ValueError("Data must be a 2D numpy array with two columns")
        if data.shape[0] < 2:
            raise ValueError("Data must have at least two rows")
        if np.any(np.isnan(data)):
            raise ValueError("Data contains NaN values")
    
        
        # Estimate the resonance frequency (f0) as the frequency at the minimum amplitude (dip)
        index_dip = np.argmin(data[:, 1])
        fmin = data[index_dip, 0]  # fmin is the frequency at the minimum amplitude (dip)
        self.params["fmin"] = fmin
        self.params["f0"] = 0 # Shifted to about the center in the model function

        # Estimate the baseline from data endpoints (first and last 10% of points)
        N = data.shape[0]
        edge_count = max(1, int(0.1 * N))
        baseline_left = np.median(data[:edge_count, 1])
        baseline_right = np.median(data[-edge_count:, 1])
        baseline = (baseline_left + baseline_right) / 2

        fwhm = self._peak_width(data[:, 0], data[:, 1])
        if fwhm <= 0:
            # Fallback to using a fraction of the frequency range if a valid FWHM cannot be found
            fwhm = 0.05 * (np.max(data[:, 0]) - np.min(data[:, 0]))

        # Estimate total Q: Q_t ~ f0 / fwhm
        Qt = fmin / fwhm if fwhm != 0 else (29208/1.5)
        self.params["Qt"] = Qt
        
        # Qc is estimated as Qc = 1.1 * Qt
        Qc = 1.1 * Qt 
        self.params["Qc"] = Qc

        # Estimate the phase shift (phi) as a small negative value
        self.params["phi"] = 0

        # Set baseline polynomial parameters: A is baseline, B, C, D are 0
        self.params["A"] = baseline
        self.params["B"] = 2.8643e-8
        self.params["C"] = 8.0398e-15
        self.params["D"] = -3.5988e-20

        # Compute the amplitude factor K.
        K = (np.max(data[:, 1]) - np.min(data[:, 1])) * Qc / Qt if Qt != 0 else 0.0
        self.params["K"] = K

        # Assign the found parameters to the model
        self._validate_search()
        model.assing_params(self.params)

        # Fix fmin, it will be used as a parameter in the model
        model.set_fixed_params(["fmin"])

    def _peak_width(self, datax, datay):
        half_height_value = np.min(datay) + (np.max(datay) - np.min(datay)) / np.sqrt(2)
        hits = []
        above = datay[0] > half_height_value
        for i in range(1, len(datay)):
            new_above = datay[i] > half_height_value
            if new_above != above: 
                hits.append((datax[i] + datax[i-1]) / 2)
                above = new_above
        return abs(hits[-1] - hits[0])