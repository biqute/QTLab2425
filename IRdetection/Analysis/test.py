import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.util import describe
import sys

def resonance_model(p, x, fmin):
    """
    Compute the resonance profile Y0 based on parameters p and frequency x.

    Parameters:
    - p: list or array of parameters [p1, p2, p3, p4, p5, p6, p7, p8]
        p1: Multiplicative factor
        p2: Qt
        p3: Qi
        p4: Rotation (in radians)
        p5: f_riso (Resonance frequency)
        p6: Linear background
        p7: Quadratic background
        p8: Cubic background
    - x: Frequency data
    - fmin: Minimum frequency value (used for normalization)

    Returns:
    - Y0: Resonance profile
    """
    p1, p2, p3, p4, p5, p6, p7, p8 = p
    term1 = p6 * x
    term2 = p7 * x**2
    term3 = p8 * x**3
    denominator = 1 + 2j * p2 * (x - p5) / fmin
    term4 = p1 * (1 - np.exp(1j * p4) * p2 * (1/p2 - 1/p3) / denominator)
    Y0 = np.abs(term1 + term2 + term3 + term4)
    return Y0

def chi_square(p, x, y, dy, fmin):
    """
    Compute the chi-square between the data y and the model Y0.

    Parameters:
    - p: list or array of parameters [p1, p2, p3, p4, p5, p6, p7, p8]
    - x: Frequency data
    - y: Measured data
    - dy: Uncertainties in the measured data
    - fmin: Minimum frequency value (used for normalization)

    Returns:
    - chi2: Chi-square value
    """
    Y0 = resonance_model(p, x, fmin)
    chi2 = np.sum(((y - Y0) / dy)**2)
    return chi2

def fit_res_db(name_file, guess_param, start=1, stop=None):
    """
    Fit the resonance data using the iminuit library.

    Parameters:
    - name_file: Path to the data file. The file should have two or three columns:
        1. Frequency
        2. S21[dB]
        3. (Optional) Delta_Y (uncertainties)
    - guess_param: Initial guess for the parameters [p1, p2, p3, p4, p5, p6, p7, p8].
    - start: Starting index for fitting (1-based index). Default is 1.
    - stop: Stopping index for fitting. Default is the last data point.

    Returns:
    - fitted_params: Fitted parameters after optimization.
    """
    # Load data from file
    try:
        data = np.loadtxt(name_file, delimiter=' ', skiprows=0)
    except Exception as e:
        raise IOError(f"Error reading the file {name_file}: {e}")

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    num_columns = data.shape[1]
    if num_columns < 2:
        raise ValueError("Data file must contain at least two columns: frequency and S21[dB].")

    x_full = data[:, 0]
    y_full_dB = data[:, 1]

    # Set default stop if not provided
    if stop is None:
        stop = len(x_full)
    else:
        stop = min(stop, len(x_full))

    # Adjust for zero-based indexing in Python
    start_idx = start - 1 if start > 0 else 0
    stop_idx = stop

    # Slice the data
    x = x_full[start_idx:stop_idx]
    y_dB = y_full_dB[start_idx:stop_idx]

    # Convert S21 from dB to linear scale
    y = 10**(y_dB / 20)

    # Normalize y by the mean of the first 10 points
    num_points_to_average = min(10, len(y))
    y_mean = np.mean(y[:num_points_to_average])
    y_normalized = y / y_mean

    # Determine uncertainties
    if num_columns >= 3:
        dy = data[start_idx:stop_idx, 2]
    else:
        dy = np.ones_like(y_normalized)

    # Find the minimum of y to determine fmin
    I = np.argmin(y_normalized)
    fmin = x[I]

    # Define parameter limits
    param_limits = {
        'p1': (0.8 * guess_param[0], 1.2 * guess_param[0]),               # Multiplicative factor
        'p2': (1e2, 1e7),               # Qt
        'p3': (1e3, 1e7),               # Qi
        'p4': (-np.pi, np.pi),          # Rotation
        'p5': (x.min(), x.max()),       # f_riso
        'p6': (-1e-8, 1e-8),            # Linear background
        'p7': (-1e-17, 1e-17),          # Quadratic background
        'p8': (-1e-22, 1e-22)           # Cubic background
    }

    # Initialize Minuit with the chi-square function
    def chi2_wrapper(p1, p2, p3, p4, p5, p6, p7, p8):
        params = [p1, p2, p3, p4, p5, p6, p7, p8]
        return chi_square(params, x, y_normalized, dy, fmin)

    # Create the Minuit object
    m = Minuit(
        chi2_wrapper,
        p1=guess_param[0],
        p2=guess_param[1],
        p3=guess_param[2],
        p4=guess_param[3],
        p5=guess_param[4],
        p6=guess_param[5],
        p7=guess_param[6],
        p8=guess_param[7]
    )

    # Apply parameter limits
    for param, (lower, upper) in param_limits.items():
        m.limits[param] = (lower, upper)

    # Perform the minimization
    m.migrad()

    # Check if the fit was successful
    if not m.valid:
        raise RuntimeError("Minimization failed. Check the initial parameters and data.")

    # Extract fitted parameters
    fitted_params = [
        m.values['p1'],
        m.values['p2'],
        m.values['p3'],
        m.values['p4'],
        m.values['p5'],
        m.values['p6'],
        m.values['p7'],
        m.values['p8']
    ]

    # Compute the fitted resonance profile
    Y0_fit = resonance_model(fitted_params, x, fmin)

    # Plot the data and the fit
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_normalized, 'b.', label='Data')
    plt.plot(x, Y0_fit, 'r-', label='Fit')
    plt.xlabel('Frequency')
    plt.ylabel('S21 (Normalized)')
    plt.title('Resonance Fit')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Display fitted parameters
    Q_t = fitted_params[1]
    Q_i = fitted_params[2]
    if Q_i != Q_t:
        Q_c = (Q_t * Q_i) / (Q_i - Q_t)
    else:
        Q_c = np.inf  # Avoid division by zero
    f_riso = fitted_params[4] + fmin

    print(f"Q_t = {Q_t:.4e}")
    print(f"Q_i = {Q_i:.4e}")
    print(f"Q_c = {Q_c:.4e}")
    print(f"f_riso = {f_riso:.4f}")

    return fitted_params

# Example usage:
if __name__ == "__main__":
    # Define the initial guess parameters
    # [p1, p2, p3, p4, p5, p6, p7, p8]
    guess_parameters = [1, 1e3, 6e4, -0.5, 0, -1e-9, -1e-19, 0]

    # Path to your data file
    data_file = 'Data/fit_test/Resonances/20mK_-5dBm.txt'  # Replace with your actual file path

    # Define start and stop indices
    start_index = 1
    stop_index = 1601

    # Perform the fit
    try:
        fitted_parameters = fit_res_db(data_file, guess_parameters, start=start_index, stop=stop_index)
    except Exception as e:
        print(f"An error occurred during fitting: {e}", file=sys.stderr)
