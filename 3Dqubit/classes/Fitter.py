import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy import stats

import sys; sys.path.append("../utils")
from number_to_text import number_to_text 


class Fitter():
    '''
    Fitter
    '''

    datax = np.array([])
    datay = np.array([])
    sigmay = np.array([])
    model = None
    
    params = {}
    param_units = {}
    param_displayed_names = {}
    
    title = "Title"
    labelx = "x"
    labely = "y"
    unitx = ""
    unity = ""

    number_of_errorbars = 50

    def fit(self):
        params_values = {}
        params_limits = {}
        for key, value in self.params.items():
            params_values[key] = value[1]
            params_limits[key] = (value[0], value[2])

        least_squares = LeastSquares(self.datax, self.datay, self.sigmay, self.model)
        m = Minuit(least_squares, **params_values) 
        
        for key, value in params_limits.items(): m.limits[key] = value
        
        m.migrad()
        m.hesse()

        final_params = {}
        for p, v, e in zip(m.parameters, m.values, m.errors):
            final_params[p] = {"value": v, "sigma": e}

        return {
            "chi2": m.fval,
            "ndof": m.ndof,
            "reduced_chi2": m.fmin.reduced_chi2,
            "pvalue": 1 - stats.chi2.cdf(m.fval, m.ndof),
            "params": final_params,
        }
    
    def plot_fit(self):
        res = self.fit()
        final_values = {}
        for key, value in res["params"].items():
            final_values[key] = value["value"]

        fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})
        first = axes[0]
        second = axes[1]

        plt.subplots_adjust(hspace = 0.05)

        # MAIN PLOT
        modelx = np.linspace(np.min(self.datax), np.max(self.datax), 2000)
        modely = [self.model(f, **final_values) for f in modelx]

        first.plot(self.datax, self.datay, color = "blue", label="data")
        first.plot(modelx, modely, color = "red", label="model")
        first.set_title(self.title)

        # Box
        text = ""
        for key, par in res["params"].items():
            name = self.param_displayed_names[key] if key in self.param_displayed_names else key
            text += rf"${name}$ = ${number_to_text(par["value"], par["sigma"])}${self.param_units[key]}" + "\n"
        text += f"p-value = {res["pvalue"]:.2f}"
        first.text(
            np.min(self.datax) + (np.max(self.datax) - np.min(self.datax))*0.02, 
            np.min(self.datay) + (np.max(self.datay) - np.min(self.datay))*0.02, 
            text, 
            fontsize = 12, 
            bbox = dict(facecolor='white', edgecolor='black', boxstyle='square, pad=0.5')
        )

        # Axis
        first.set(ylabel=f"{self.labely} [{self.unity}]")
        first.set_xticks(np.linspace(np.min(self.datax), np.max(self.datax), 25))
        first.set_xlim(np.min(self.datax), np.max(self.datax))
        first.set_xticklabels([''] * 25)
        first.grid(linestyle='--', linewidth=0.5)
        legend = first.legend(loc="upper right", frameon=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_alpha(1.0)
        legend.get_frame().set_boxstyle("Square")

        # RESIDUALS PLOT
        residualsy = [ self.datay[i] - self.model(self.datax[i], **final_values) for i in range(len(self.datax)) ]
        zeroy = np.zeros(len(self.datax))
        N = int(len(self.datax) / self.number_of_errorbars)

        second.errorbar(
            self.datax[::N], 
            residualsy[::N], 
            yerr = self.sigmay[::N],
            ecolor = "lightblue",
            capsize = 5,
            fmt = '',
            linestyle=''
        )
        second.plot(self.datax, zeroy, color = "red")
        second.plot(self.datax, residualsy, color = "blue")

        # Axis
        second.set_xlim(np.min(self.datax), np.max(self.datax))
        second.set(
            xlabel=f"{self.labelx} [{self.unitx}]",
            ylabel=f"Residuals [{self.unity}]"
        )
        second.set_xticks(np.linspace(np.min(self.datax), np.max(self.datax), 25))
        plt.xticks(rotation=90)
        second.grid(linestyle='--', linewidth=0.5)

        plt.show()

    
