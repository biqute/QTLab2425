import numpy as np
from peak_width import peak_width

def estimate_rettaroli_params(freqs, S21, S11):
    """
    estimate_rettaroli_params(freqs, S21, S11)

    Returns a dictionary with the following keys:
    - "f0": resonance frequency
    - "QL": loaded quality factor
    - "k1": coupling coefficient to the first port
    - "k2": coupling coefficient to the second port
    - "A": amplification factor
    - "B": parameter B
    - "C": parameter C
    - "theta11": phase of $S_{11}$ at resonance
    - "theta21": phase of $S_{21}$ at resonance
    - "width": width of the resonance peak

    Estimates parameters for the models

    $ S_{11}(f) = Ae^{j\theta_{11}} \frac{B(k_1, k_2) + jQ_L\delta(f, f_0)}{1 + jQ_L\delta(f, f_0)}$

    $ S_{21}(f) = Ae^{j\theta_{21}}\frac{C(k_1, k_2)}{1 + jQ_L\delta(f, f_0)}$

    with

    $ B(k_1, k_2) = \frac{k_1 - k_2 - 1}{k_1 + k_2 + 1}$

    $ C(k_1, k_2) = \frac{2\sqrt{k_1k_2}}{1 + k_1 + k_2}$

    """
    
    # Given a width in the same units as datax and center value, returns the indices of the datax array that correspond to the range [center - width/2, center + width/2].
    def width_to_indeces(width, center, datax):  
        deltaf = ( datax[-1] - datax[0] ) / len(datax)
        min_f = center - width / 2.0
        max_f = center + width / 2.0
        min_f_idx = int((min_f - datax[0]) / deltaf)
        max_f_idx = int((max_f - datax[0]) / deltaf)
        min_f_idx = min(len(datax) - 1, max(0, min_f_idx))
        max_f_idx = min(len(datax) - 1, max(0, max_f_idx))

        if min_f_idx == max_f_idx:
            raise ValueError(f"min_index = max_index = {min_f_idx}, no data in this range")
        if min_f_idx > max_f_idx:
            raise ValueError("min_index > max_index, something went wrong")

        return (min_f_idx, max_f_idx)

    f0 = freqs[np.argmax(np.abs(S21))]
    width = peak_width(freqs, np.abs(S21)) # no - in front of datay

    QL = f0 / width # loaded quality factor

    theta21 = np.angle(S21)[np.argmax(np.abs(S21))] # Arg[S21(w_0)]
    theta11_up_to_pi = np.angle(S11)[np.argmax(np.abs(S21))] # Arg[S11(w_0)] up to pi due to sign of B

    (id_phi, _) = width_to_indeces(width / 3, f0, freqs)
    id_phi = np.argmax(np.abs(S21)) - max(1, np.argmax(np.abs(S21)) - id_phi) # we move at least one datapoint to the left
    phi = np.angle(S11[id_phi] / np.exp(1j * theta11_up_to_pi))
    f_phi = freqs[id_phi]
    d_phi = f_phi/f0 - f0/f_phi
    B = (1 - np.tan(phi)*d_phi*QL) / (1 + np.tan(phi)/d_phi/QL)

    theta11 = theta11_up_to_pi
    if B < 0: theta11 = np.angle(np.exp(1j * (theta11_up_to_pi - np.pi))) # remove pi and make sure you are in [-pi, pi]

    S21_resonance = np.max(np.abs(S21))
    S11_resonance = np.abs(S11)[np.argmax(np.abs(S21))]
    
    A = S11_resonance / np.abs(B)
    C = S21_resonance / A

    alpha = (1 - B)/(1 + B)
    k1 = 4 / (4*alpha - (C*(alpha + 1))**2)
    k2 = alpha*k1 - 1

    return {
        "f0": f0,
        "QL": QL,
        "k1": k1,
        "k2": k2,
        "A": A,
        "B": B,
        "C": C,
        "theta11": theta11,
        "theta21": theta21,
        "width": width
    }