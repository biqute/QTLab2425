import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from iminuit.cost import LeastSquares

from FitAPI import Model, Fitter, ResonancePeakSearcher
import models as md

data_dict = {}       # stores the dataset values as NumPy arrays
metadata_dict = {}   # stores the metadata (attributes) for each dataset

def process_dataset(name, obj):
    if isinstance(obj, h5py.Dataset):
        bias_key = float(obj.attrs.get("bias_current (uA)"))
        if bias_key is not None:
            data_dict[bias_key] = obj[()]
            metadata_dict[bias_key] = {k: v for k, v in obj.attrs.items()}
        else:
            print(f"Warning: Dataset '{name}' missing 'bias_current (uA)' attribute.")

with h5py.File('Experiments/TRswipe/run-6/data/peaks_data.h5', 'r') as f:
    f.visititems(process_dataset)


bias_current = 0.5
peak_data = data_dict[bias_current]
# transpose the data to match the expected shape (rows, columns)
# peak_data = peak_data.T
f = peak_data[0]
I = peak_data[1]
Q = peak_data[2]


amp = np.sqrt(I**2 + Q**2)
amp_dBm = 10 * np.log10(amp / 1e-3)  # Convert to dBm
data_test = np.array([f, amp_dBm]).T

# try to fit
model = md.resonance_model
fitter = Fitter(model_function=md.resonance_model, 
                    param_names=["f0", "phi", "Qt", "Qc", "A", "B", "C", "D", "K", "fmin"], 
                    data=data_test, 
                    loss_function=LeastSquares
                )

searcher = ResonancePeakSearcher()
result = fitter.fit_magicus(searcher=searcher)