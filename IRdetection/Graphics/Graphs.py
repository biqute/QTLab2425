import numpy as np
import matplotlib.pyplot as plt

class fitPlotter:
    def __init__(self,
                 fit_result,
                 datax,
                 datay,
                 fit_function):
        self.fit_result = fit_result
        self.datax = datax
        self.datay = datay
        self.fit_function = fit_function

    def simple_plot(self, **kwargs):
        # Plots the data and the fit
        plt.scatter(self.datax, self.datay, label="Data", s=10)
        plt.plot(self.datax, self.fit_function(self.datax, **self.fit_result.values.to_dict()), label="Fit", color="red")
        plt.legend()
        plt.show()