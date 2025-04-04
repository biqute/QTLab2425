# Example usage
from iminuit.cost import LeastSquares
from FitAPI import Fitter, Model
import numpy as np
import numpy as np
import sys
sys.path.append('../Graphics')
from Graphics.Graphs import fitPlotter
import matplotlib.pyplot as plt
import models as md
import h5py

def S21_model(f: np.ndarray, f0: float, phi: float, Qt: float, Qc: float, A: float, B: float, C: float, D: float, K: float, fmin: float) -> np.ndarray:
    return (A+B*(f-fmin) + C*(f-fmin)**2 + D*(f-fmin)**3) + K * np.abs((1 - (Qt/np.abs(Qc))*np.exp(1j*phi)/(1 + 2j*Qt*((f-fmin) - f0)/fmin)))

# Estimate the initial guess of Qt
def peak_width(datax, datay):
    half_height_value = np.min(datay) + (np.max(datay) - np.min(datay)) / np.sqrt(2)
    hits = []
    above = datay[0] > half_height_value
    for i in range(1, len(datay)):
        new_above = datay[i] > half_height_value
        if new_above != above: 
            hits.append((datax[i] + datax[i-1]) / 2)
            above = new_above
    return abs(hits[-1] - hits[0])

result_fr = []
err_fr = []
result_Qi = []
err_Qi = []

f = []
y = []
path = 'Experiments/ResonatorsExperiment/run-4/data/peaks_data.h5'

with h5py.File(path, 'r') as hf:
    for pk in range(1, 5):
        for pw in range(-45, 0, 5):
            name = f'peak_{pk}/peak_{pk}_{pw}_dBm'
            if name in hf:
                dataset = hf[name][()]
                f.append(dataset[0 , :])  # Frequency data
                I = dataset[1 , :]  # I data
                Q = dataset[2 , :]  # Q data
                # Convert to module
                y.append(np.sqrt(I**2 + Q**2))
            else:
                print(f"Dataset {name} not found in the HDF5 file")

for pk in range(4):
    for pw in range(9):
        plt.plot(f[pw + 9*pk], y[pw + 9*pk], label=f'Peak {pk+1} Power {pw*5-45} dBm')
    plt.legend()
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('S21 (dBm)')
    plt.title('S21 vs Frequency')
    plt.grid()
    plt.show()
    plt.close()

#empty result and err arrays
result_fr = []
err_fr = []
result_Qi = []
err_Qi = []

initial_guess = {
    "f0": 17000, #0.1,
    "phi": -0.2369, # 2.1,
    "Qc": 29208,
    "A": 0.0, #1.0,
    "B": 2.8643e-8 ,#1e-9,
    "C": 8.0398e-15,#1e-18,
    "D": -3.5988e-20, # 1e-27
    }

param_limits = {
    "Qc": (1e2, 1e7),
    "Qt": (1e2, 1e7),
    #'phi': (-20, 20),
    #'f0': (-1e6, 1e6),
    }

for pk in range(4):
    for pw in range(9):
        f_i = f[int(pw + 9*pk)]
        y_i = y[int(pw + 9*pk)]
        
        fmin = f_i[np.argmin(y_i)]
        initial_guess['fmin'] = fmin
        fwhm = peak_width(f_i, -y_i)
        
        mask = (f_i > fmin - 7 * fwhm) & (f_i < fmin + 7 * fwhm)
        f_i = f_i[mask]
        y_i = y_i[mask]

        # Estimate the initial guess of Qt
        Qt_guess = fmin / peak_width(f_i, -y_i)
        initial_guess['Qt'] = Qt_guess

        # Estimate the initial guess of K
        initial_guess['K'] = (np.max(y_i) - np.min(y_i)) * initial_guess['Qc'] / initial_guess['Qt']
        #print(f"Initial guess: {initial_guess}")

        fit_data = np.column_stack((f_i, y_i))
        fitter = Fitter(model_function=md.resonance_model, 
                        param_names=["f0", "phi", "Qt", "Qc", "A", "B", "C", "D", "K", "fmin"], 
                        data=fit_data, 
                        loss_function=LeastSquares,
                        params_initial_guess=initial_guess,
                        params_range=param_limits)

        fitter.model.set_fixed_params({"fmin": fmin})
        result = fitter.fit()
        result_dict = result.values.to_dict()
        err_dict = result.errors.to_dict()
        
        result_fr.append(result_dict['f0'] + fmin)
        err_fr.append(err_dict['f0'])

        #add fmin value to result_dict
        result_dict['fmin'] = fmin
        
        result_Qi.append(abs((1/result_dict['Qt']-1/result_dict['Qc'])**-1))
        Qt = result_dict['Qt']
        Qc = result_dict['Qc']
        err_Qi.append(np.sqrt((err_dict['Qt'] * Qt**-2)**2 + (err_dict['Qc'] * Qc**-2)**2) * ((1/Qt-1/Qc)**-2))
        
        # print("Q_res" + str(i*2 + 40))
        # print("Q_i: " + str(abs((1/result_dict['Qt']-1/result_dict['Qc'])**-1)))
        
        # print(f"Result: {result_dict}")
        # grapher = fitPlotter(result_dict, f_i, y_i, md.resonance_model)
        # grapher.simple_plot()

        plt.scatter(f_i, y_i, label=f'Data - Pk: {pk+1}, Pw: {pw*5-45} dBm', s=10)
        plt.plot(f_i, md.resonance_model(f_i, **result_dict), label=f'Fit - Pk: {pk+1}, Pw: {pw*5-45} dBm')
    plt.legend()
    plt.show()

print(result_fr)
print(result_Qi)
    
