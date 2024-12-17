import numpy as np

import numpy as np

def number_to_text(x, sx, significant_figures=None):
    """
    number_to_text(x, sx, significant_figures=None)

    Parameters:
      - x is a number
      - sx is its uncertainty
      - significant_figures (optional) is the number of significant digits 
    """

    og = int(np.floor(np.log10(abs(x))))  # order of magnitude

    ogs = int(np.floor(np.log10(abs(sx))))
    sx = round(sx / (10 ** ogs)) * (10 ** ogs)

    if significant_figures is None or significant_figures <= 0:
        # correctly use the rounded version of sx
        significant_figures = max(1, 1 + og - int(np.floor(np.log10(abs(sx)))))

    significant_figures -= 1

    mantissa_x = f"{x / (10 ** og):.{significant_figures}f}"
    mantissa_sx = f"{sx / (10 ** og):.{significant_figures}f}"

    if og == 0:
        text = f"({mantissa_x} ± {mantissa_sx})"
    else:
        text = f"({mantissa_x} ± {mantissa_sx}) × 10^{{{og}}}"

    return text