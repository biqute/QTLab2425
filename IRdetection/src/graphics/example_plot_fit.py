"""
Example usage of the improved plot_fit function.
"""

import numpy as np
import matplotlib.pyplot as plt
from plots import plot_fit
from colours import Palette
from typography import Typography, Font

# Generate some example data with noise
def true_function(x):
    """Example function: quadratic with some complexity"""
    return 2 * x**2 - 3 * x + 1 + 0.5 * np.sin(5 * x)

def fitted_model(x):
    """Simplified model that doesn't capture all complexity"""
    return 2 * x**2 - 3 * x + 1

# Generate noisy data
np.random.seed(42)
x_data = np.linspace(-2, 3, 50)
y_true = true_function(x_data)
noise = np.random.normal(0, 0.3, len(x_data))
y_data = y_true + noise

# Create data dictionary
data = {'x': x_data, 'y': y_data}

palette = Palette("Science Professional", {
    'primary': "#239E25",      # Deep blue - main data
    'secondary': '#2A9D8F',    # Teal - secondary data
    'accent': "#A74444",       # Amber - highlights
    'accent2': '#F4A261',      # Coral - secondary highlights
    'background': '#E6EBE0',   # Clean white background
    'background2': '#F8F9FA',  # Light gray secondary background
    'text_primary': '#264653', # Dark blue-gray for text
    'text_secondary': '#2A9D8F', # Medium teal for secondary text
    'status_error': '#E76F51',  # Coral red for errors/warnings
    'neutral_light': "#E7E7E7",  # Light neutral for grid
})

typography = Typography(
    title=Font('Palatino Linotype', 18, family='serif'),        # Elegant serif for titles
    subtitle=Font('Verdana', 15, family='serif'),     # Consistent serif for subtitles
    body=Font('Verdana', 12, family='sans-serif'),          # Clean sans-serif for readability
    caption=Font('Verdana', 10, family='sans-serif')        # Smaller Verdana for captions
)

# Create the plot with default matplotlib/seaborn styling
print("Creating plot with default matplotlib styling...")
fig1, axes1 = plot_fit(
    data=data, 
    model=fitted_model,
    title='Quadratic Model Fit with Default Styling',
    X_label='X Values',
    Y_label='Y Values',
    show_residuals=True,
    figsize=(12, 10),
    palette=palette,
    typography=typography,
)

# Save the default plot
# plt.savefig('example_fit_plot_default.png', dpi=300, bbox_inches='tight')
plt.show()

# You can also create a plot with custom palette and typography if available
print("Plot with default styling saved as 'example_fit_plot_default.png'")
print(f"Main plot axis: {axes1[0]}")
print(f"Residuals plot axis: {axes1[1]}")
