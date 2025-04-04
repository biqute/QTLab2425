import numpy as np
import matplotlib.pyplot as plt

class fitPlotter:
    def __init__(self,
                 fit_result_dict,
                 datax,
                 datay,
                 fit_function,
                 err_y=None):
        self.fit_result_dict = fit_result_dict
        self.datax = datax
        self.datay = datay
        self.err_y = err_y
        self.fit_function = fit_function

    def simple_plot(self, **kwargs):
        # Plots the data and the fit
        if self.err_y is not None:
            plt.errorbar(self.datax, self.datay, yerr=self.err_y, fmt='o', label="Data")
        else:
            plt.scatter(self.datax, self.datay, label="Data", s=10)
        
        plt.plot(self.datax, self.fit_function(self.datax, **self.fit_result_dict), label="Fit", color="red")
        plt.legend()
        plt.show()
