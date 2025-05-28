"""
Example plot generators for testing and showcasing Palette and Typography combinations.

This module provides utility functions to generate various types of plots with synthetic
data to demonstrate how different styling choices affect plot appearance. All generated
data is deterministic (same seed) for consistent comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Tuple, Optional
import seaborn as sns

from typography import Typography
from colours import Palette
from plotting import set_plotting_style


class PlotExamples:
    """
    A class for generating example plots to showcase Palette and Typography combinations.
    
    This class provides methods to create various types of plots with synthetic data,
    allowing users to preview how their styling choices will look in practice.
    
    Parameters
    ----------
    typography : Typography
        Typography settings to use for all example plots.
    palette : Palette
        Color palette to use for all example plots.
    style_type : str, optional
        Style type to apply ('publication', 'presentation', 'notebook', 'dark').
        Default is 'publication'.
    
    Attributes
    ----------
    typography : Typography
        The typography settings being used.
    palette : Palette
        The color palette being used.
    style_type : str
        The style type being applied.
    rng : numpy.random.Generator
        Random number generator with fixed seed for reproducible data.
    """
    
    def __init__(self, typography: Typography, palette: Palette, style_type: str = 'publication'):
        self.typography = typography
        self.palette = palette
        self.style_type = style_type
        # Fixed seed for reproducible synthetic data
        self.rng = np.random.default_rng(42)
        
        # Apply the plotting style
        set_plotting_style(typography, palette, style_type)
    
    def line_plot_example(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Generate an example line plot with multiple series.
        
        Creates a line plot showing synthetic time series data with multiple curves,
        demonstrating color cycling, line styles, and legend formatting.
        
        Parameters
        ----------
        figsize : tuple of int, optional
            Figure size as (width, height). Default is (10, 6).
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Generate synthetic time series data
        x = np.linspace(0, 10, 100)
        data_series = {
            'Signal A': np.sin(x) + 0.1 * self.rng.normal(0, 1, 100),
            'Signal B': np.cos(x * 1.2) + 0.15 * self.rng.normal(0, 1, 100),
            'Signal C': np.sin(x * 0.8) * np.exp(-x/8) + 0.1 * self.rng.normal(0, 1, 100),
            'Baseline': 0.2 * np.sin(x * 0.3) + 0.05 * self.rng.normal(0, 1, 100)
        }
        
        for label, y_data in data_series.items():
            ax.plot(x, y_data, label=label, linewidth=2.5, alpha=0.8)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (V)')
        ax.set_title('Multi-Channel Signal Analysis')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def scatter_plot_example(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Generate an example scatter plot with different categories.
        
        Creates a scatter plot with synthetic data points in different categories,
        demonstrating color mapping, marker styles, and statistical visualization.
        
        Parameters
        ----------
        figsize : tuple of int, optional
            Figure size as (width, height). Default is (10, 6).
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Generate synthetic experimental data
        n_points = 150
        categories = ['Control', 'Treatment A', 'Treatment B', 'Treatment C']
        
        for i, category in enumerate(categories):
            # Generate clustered data for each category
            x_center = 2 + i * 1.5
            y_center = 3 + i * 0.8
            
            x = self.rng.normal(x_center, 0.6, n_points // len(categories))
            y = self.rng.normal(y_center, 0.4, n_points // len(categories))
            
            # Add some correlation
            y += 0.3 * (x - x_center)
            
            ax.scatter(x, y, label=category, alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
        
        ax.set_xlabel('Parameter X (units)')
        ax.set_ylabel('Response Y (units)')
        ax.set_title('Experimental Results: Treatment Effects')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def bar_chart_example(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Generate an example bar chart with error bars.
        
        Creates a grouped bar chart with synthetic measurement data,
        demonstrating categorical data visualization with uncertainty.
        
        Parameters
        ----------
        figsize : tuple of int, optional
            Figure size as (width, height). Default is (10, 6).
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Synthetic measurement data
        categories = ['Sample A', 'Sample B', 'Sample C', 'Sample D', 'Sample E']
        methods = ['Method 1', 'Method 2', 'Method 3']
        
        # Generate data with realistic relationships
        base_values = [85, 92, 78, 95, 89]
        data = {}
        errors = {}
        
        for i, method in enumerate(methods):
            multiplier = 1.0 + (i - 1) * 0.15  # Methods have different scales
            noise_level = 3 + i * 2  # Different uncertainty levels
            
            values = [val * multiplier + self.rng.normal(0, noise_level) for val in base_values]
            error_vals = [noise_level + self.rng.uniform(1, 3) for _ in base_values]
            
            data[method] = values
            errors[method] = error_vals
        
        x = np.arange(len(categories))
        width = 0.25
        
        for i, method in enumerate(methods):
            offset = (i - 1) * width
            ax.bar(x + offset, data[method], width, label=method, 
                  yerr=errors[method], capsize=5, alpha=0.8)
        
        ax.set_xlabel('Sample Groups')
        ax.set_ylabel('Measurement Value (units)')
        ax.set_title('Comparative Analysis: Multiple Methods')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def heatmap_example(self, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Generate an example heatmap with correlation data.
        
        Creates a heatmap showing synthetic correlation matrix data,
        demonstrating color mapping for matrix visualization.
        
        Parameters
        ----------
        figsize : tuple of int, optional
            Figure size as (width, height). Default is (10, 8).
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Generate synthetic correlation matrix
        variables = ['Temperature', 'Pressure', 'Humidity', 'Wind Speed', 
                    'Solar Rad.', 'Rainfall', 'Air Quality', 'Visibility']
        n_vars = len(variables)
        
        # Create a realistic correlation matrix
        base_matrix = self.rng.uniform(-0.3, 0.3, (n_vars, n_vars))
        
        # Add some structure to make it more realistic
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    base_matrix[i, j] = 1.0
                elif abs(i - j) == 1:
                    base_matrix[i, j] = self.rng.uniform(0.3, 0.7)
                    base_matrix[j, i] = base_matrix[i, j]
                elif abs(i - j) <= 2:
                    base_matrix[i, j] = self.rng.uniform(-0.2, 0.4)
                    base_matrix[j, i] = base_matrix[i, j]
        
        # Use seaborn for better heatmap styling
        sns.heatmap(base_matrix, annot=True, fmt='.2f', 
                   xticklabels=variables, yticklabels=variables,
                   cmap='RdBu_r', center=0, square=True, 
                   cbar_kws={'label': 'Correlation Coefficient'}, ax=ax)
        
        ax.set_title('Environmental Variables Correlation Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        return fig
    
    def time_series_example(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Generate an example time series plot with date formatting.
        
        Creates a time series plot with synthetic daily data over several months,
        demonstrating date formatting and trend visualization.
        
        Parameters
        ----------
        figsize : tuple of int, optional
            Figure size as (width, height). Default is (12, 6).
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Generate synthetic time series data
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 6, 30)
        dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days)]
        
        # Create realistic-looking data with trends and noise
        t = np.arange(len(dates))
        
        # Seasonal component
        seasonal = 20 * np.sin(2 * np.pi * t / 365.25) + 15 * np.sin(2 * np.pi * t / 30.44)
        
        # Trend component
        trend = 0.02 * t + 50
        
        # Noise
        noise = self.rng.normal(0, 5, len(dates))
        
        # Combine components
        values = trend + seasonal + noise
        
        # Add some outliers
        outlier_indices = self.rng.choice(len(dates), size=10, replace=False)
        values[outlier_indices] += self.rng.normal(0, 20, 10)
        
        ax.plot(dates, values, linewidth=2, alpha=0.8, label='Measured Data')
        
        # Add trend line
        z = np.polyfit(t, values, 1)
        p = np.poly1d(z)
        ax.plot(dates, p(t), '--', linewidth=2, alpha=0.7, label='Linear Trend')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Long-term Temperature Monitoring')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig
    
    def subplot_example(self, figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        Generate an example multi-subplot figure.
        
        Creates a figure with multiple subplots showing different types of data,
        demonstrating layout management and consistency across plot types.
        
        Parameters
        ----------
        figsize : tuple of int, optional
            Figure size as (width, height). Default is (14, 10).
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Subplot 1: Simple line plot
        x1 = np.linspace(0, 4*np.pi, 100)
        y1 = np.sin(x1) * np.exp(-x1/8)
        ax1.plot(x1, y1, linewidth=2.5)
        ax1.set_title('Damped Oscillation')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Histogram
        data2 = self.rng.normal(100, 15, 1000)
        ax2.hist(data2, bins=30, alpha=0.7, edgecolor='white', linewidth=0.5)
        ax2.set_title('Distribution Analysis')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Box plot
        data3 = [self.rng.normal(0, std, 100) for std in range(1, 6)]
        bp = ax3.boxplot(data3, patch_artist=True)
        
        # Color the boxes with palette colors
        colors = [str(color) for color in [
            self.palette.colours.get('primary', '#1f77b4'),
            self.palette.colours.get('secondary', '#ff7f0e'),
            self.palette.colours.get('accent', '#2ca02c'),
            self.palette.colours.get('accent2', '#d62728'),
            self.palette.colours.get('status_info', '#9467bd')
        ]]
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.set_title('Variability Comparison')
        ax3.set_xlabel('Group')
        ax3.set_ylabel('Value')
        ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Pie chart
        sizes = [25, 30, 15, 20, 10]
        labels = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
        wedges, texts, autotexts = ax4.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                          startangle=90, colors=colors)
        ax4.set_title('Distribution Overview')
        
        plt.suptitle('Multi-Panel Analysis Dashboard', fontsize=self.typography.title.size + 2, 
                    fontweight='bold', y=0.98)
        plt.tight_layout()
        return fig
    
    def generate_all_examples(self, save_dir: Optional[str] = None) -> dict:
        """
        Generate all example plots and optionally save them.
        
        Creates all available example plots and returns them as a dictionary.
        Optionally saves the plots to the specified directory.
        
        Parameters
        ----------
        save_dir : str, optional
            Directory path to save the generated plots. If None, plots are not saved.
            
        Returns
        -------
        dict
            Dictionary with plot names as keys and figure objects as values.
        """
        examples = {
            'line_plot': self.line_plot_example(),
            'scatter_plot': self.scatter_plot_example(),
            'bar_chart': self.bar_chart_example(),
            'heatmap': self.heatmap_example(),
            'time_series': self.time_series_example(),
            'subplots': self.subplot_example()
        }
        
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            
            for name, fig in examples.items():
                filepath = os.path.join(save_dir, f'{name}_example_{self.style_type}.png')
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"Saved: {filepath}")
        
        return examples


def quick_preview(typography: Typography, palette: Palette, style_type: str = 'publication', save_path: Optional[str] = None):
    """
    Generate a quick preview of how the typography and palette combination looks.
    
    Creates a simple multi-plot figure to quickly assess the visual appeal
    of a typography and palette combination.
    
    Parameters
    ----------
    typography : Typography
        Typography settings to preview.
    palette : Palette
        Color palette to preview.
    style_type : str, optional
        Style type to apply. Default is 'publication'.
    save_path : str, optional
        Path to save the preview figure. If None, the figure is not saved.
        
    Returns
    -------
    matplotlib.figure.Figure
        A figure with multiple example plots for quick assessment.    """
    # Apply the plotting style
    set_plotting_style(typography, palette, style_type)
    
    # Create a condensed preview figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Quick line plot
    x = np.linspace(0, 10, 50)
    rng = np.random.default_rng(42)
    for i in range(3):
        y = np.sin(x + i) + 0.1 * rng.normal(0, 1, 50)
        ax1.plot(x, y, label=f'Series {i+1}', linewidth=2)
    ax1.set_title('Line Plot Preview')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Quick scatter
    x_scatter = rng.normal(5, 2, 100)
    y_scatter = rng.normal(5, 2, 100)
    ax2.scatter(x_scatter, y_scatter, alpha=0.6, s=50)
    ax2.set_title('Scatter Plot Preview')
    ax2.grid(True, alpha=0.3)
    
    # Quick bar chart
    categories = ['A', 'B', 'C', 'D']
    values = [23, 45, 56, 78]
    ax3.bar(categories, values, alpha=0.8)
    ax3.set_title('Bar Chart Preview')
    ax3.grid(True, alpha=0.3)
    
    # Quick histogram
    data = rng.normal(50, 15, 1000)
    ax4.hist(data, bins=20, alpha=0.7, edgecolor='white')
    ax4.set_title('Histogram Preview')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Style Preview: {palette.name} with {style_type.title()} Style', 
                fontsize=typography.title.size, fontweight='bold')
    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Preview saved to: {save_path}")

    return fig


if __name__ == "__main__":
    # Example usage
    from typography import Typography, Font
    from colours import Palette, COLOURS
    
    # Create example typography and palette
    typography = Typography(
        title=Font('Palatino', 16),
        subtitle=Font('Palatino', 14),
        body=Font('Verdana', 12),
        caption=Font('Verdana', 10)
    )
    
    # palette = Palette("Example", {
    #     'primary': COLOURS['blue'],
    #     'secondary': COLOURS['orange'],
    #     'accent': COLOURS['green'],
    #     'background': COLOURS['white']
    # })

    palette = Palette("Example", "ed6a5a-f4f1bb-9bc1bc-5ca4a9-e6ebe0")
    
    # Generate examples
    examples = PlotExamples(typography, palette, 'presentation')
    all_plots = examples.generate_all_examples()
    
    # Show all plots
    plt.show()
