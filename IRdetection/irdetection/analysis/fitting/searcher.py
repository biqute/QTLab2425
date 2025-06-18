import numpy as np
from abc import ABC, abstractmethod
from irdetection.analysis.fitting.models import Model


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


class ResonancePeakSearcher(Searcher):
    def __init__(self, version: str = "T"):
        super().__init__()
        self.version = version
        self.params = {
            "f0": None,
            "phi": None,
            "A": None,
            "B": None,
            "C": None,
            "D": None,
            "K": None,
            "fmin": None
        }
        if version == "T":
            self.params["Qt"] = None
            self.params["Qc"] = None
            
        elif version == "I":
            self.params["Qs"] =None
            self.params["Qi"] = None

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
        
        # Estimate the phase shift (phi) as a small negative value
        self.params["phi"] = 0

        # Set baseline polynomial parameters: A is baseline, B, C, D are 0
        self.params["A"] = baseline
        self.params["B"] = 2.8643e-8
        self.params["C"] = 8.0398e-15
        self.params["D"] = -3.5988e-20


        fwhm = self._peak_width(data[:, 0], data[:, 1])
        if fwhm <= 0:
            # Fallback to using a fraction of the frequency range if a valid FWHM cannot be found
            fwhm = 0.05 * (np.max(data[:, 0]) - np.min(data[:, 0]))

        # Estimate total Q: Q_t ~ f0 / fwhm
        Qi = fmin / fwhm if fwhm != 0 else (29208/1.5)

        #first estimation of Qc
        Qc = 2e3 # 1.1* Qt
        
        Qt = 1/(1/Qi + 1/Qc)
                
        # Compute the amplitude factor K.
        K = (np.max(data[:, 1]) - np.min(data[:, 1])) * Qc / Qt if Qt != 0 else 0.0
        self.params["K"] = K
        
        # Qc is estimated at f = fmin, needs more work
        #y = data[index_dip, 1]  # Amplitude at the dip
        #f0 = 0.1
        #A = self.params["A"]
        #Qc = K/(K - y + A) * Qt * np.sqrt(2)/2 #np.sqrt(1 + (2* Qt * f0 / fmin)**2))
        
        if self.version == "I":
            Qi = Qt*Qc/(Qc - Qt)  # Estimate Qi using the relation Qs = Qt*Qc/(Qc - Qt)
            Qs = Qi/Qc
            self.params["Qs"] = Qs
            self.params["Qi"] = Qi
            #set boundaries for Qc and Qt
            model.set_param_bounds({
                "Qi": (0, 1e6),
                "Qs": (0, 1e6)
            })
            
        elif self.version == "T":
            # For the T version, we use Qt and Qc
            self.params["Qt"] = Qt
            self.params["Qc"] = Qc
            #set boundaries for Qc and Qt
            model.set_param_bounds({
                "Qt": (0, 1e6),
                "Qc": (0, 1e6)
            })
        

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