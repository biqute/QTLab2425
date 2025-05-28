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

palette = Palette("Quantum Science", {
    'primary': "#0B4D6A",      # Deep ocean blue - sophisticated and trustworthy
    'secondary': '#1B263B',    # Midnight navy - elegant depth
    'accent': "#FF6B35",       # Vibrant coral - high-impact highlights
    'accent2': '#FFD23F',      # Golden yellow - energy and precision
    'background': '#F8F9FB',   # Ultra-clean off-white - pristine lab feel
    'background2': '#E8EDF3',  # Subtle blue-gray - sophisticated secondary
    'text_primary': '#1A1B23', # Rich charcoal - maximum readability
    'text_secondary': '#4A5568', # Cool gray - perfect for secondary text
    'status_error': '#E53E3E',  # Crisp red - clear error indication
    'neutral_light': "#E2E8F0",  # Cool light gray - subtle grid
    'neutral_dark': "#2D3748",   # Professional dark gray - strong borders
})

typography = Typography(
    title=Font('Computer Modern', 20, family='serif'),        # LaTeX-style elegance
    subtitle=Font('Helvetica Neue', 16, family='sans-serif'), # Modern Swiss precision
    body=Font('Source Sans Pro', 13, family='sans-serif'),    # Optimized for scientific reading
    caption=Font('Source Sans Pro', 11, family='sans-serif')  # Clean and consistent
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
