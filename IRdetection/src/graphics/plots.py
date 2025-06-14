import numpy as np
import matplotlib.pyplot as plt

from colours import Palette
from typography import Typography


def plot_fit(data, model, model_params, title: str = 'Model Fit', palette: Palette = None, typography: Typography = None, 
             xlabel: str = 'X-axis', ylabel: str = 'Y-axis', show_residuals: bool = True, 
             figsize: tuple = (10, 8), ax = None, rasterize_points: bool = False, **kwargs):
    """
    Plot the data and model fit with customized aesthetics, including residuals plot.

    Parameters
    ----------
    data : dict or array-like
        The data to plot. Should contain 'x' and 'y' keys if dict, or be structured array.
    model : callable
        The model function to fit the data.
    model_params : dict
        Parameters for the model function. Should match the model's expected parameters.
    title : str, optional
        The title of the plot. Default is 'Model Fit'.
    palette : Palette, optional
        The color palette for the plot.
    typography : Typography, optional
        The typography settings for the plot.    
    xlabel : str, optional
        Label for the x-axis. Default is 'X-axis'.
    ylabel : str, optional
        Label for the y-axis. Default is 'Y-axis'.
    show_residuals : bool, optional
        Whether to show residuals plot below main plot. Default is True.    figsize : tuple, optional
        Figure size (width, height). Default is (10, 8).
    ax : matplotlib.axes.Axes, optional
        Axes to subdivide into main and residual plots. If None, a new figure will be created.
    rasterize_points : bool, optional
        Whether to rasterize the scatter points for better performance with large datasets. Default is False.
    **kwargs
        Additional keyword arguments passed to plotting functions.

    Returns
    -------
    tuple
        (fig, axes) where axes is [ax_main] or [ax_main, ax_residuals] depending on show_residuals.    """    # Create or subdivide axes
    if ax is None:
        if show_residuals:
            fig, (ax_main, ax_residuals) = plt.subplots(2, 1, figsize=figsize, 
                                                      gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1})
        else:
            fig, ax_main = plt.subplots(figsize=figsize)
            ax_residuals = None
    else:
        fig = ax.get_figure()
        if show_residuals:
            # Subdivide the provided ax into main and residual plots
            pos = ax.get_position()
            ax.remove()
            
            # Create subdivided axes with correct height ratios
            ax_main = fig.add_axes([pos.x0, pos.y0 + pos.height*0.25, pos.width, pos.height*0.75])
            ax_residuals = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height*0.2])
        else:
            ax_main = ax
            ax_residuals = None
    
    # Extract data
    if isinstance(data, dict):
        x_data = data['x']
        y_data = data['y']
    else:
        # Assume it's a structured array or similar
        x_data = data[:, 0] if data.ndim > 1 else np.arange(len(data))
        y_data = data[:, 1] if data.ndim > 1 else data
      # Main plot: data and model
    scatter_color = str(palette.colours['primary']) if palette else None
    ax_main.scatter(x_data, y_data, color=scatter_color, alpha=0.3, s=30, 
                   label='Data', zorder=2, edgecolors=str(palette.primary.darken(10)), linewidth=0.8, rasterized=rasterize_points, **kwargs)
    
    # Generate smooth model curve
    x_fit = np.linspace(x_data.min(), x_data.max(), 300)
    y_fit = model(x_fit, **model_params)

    line_color = str(palette.colours['accent']) if palette else None
    ax_main.plot(x_fit, y_fit, color=line_color, linewidth=2, 
                label='Model', zorder=3, alpha=0.9)
    
    # Calculate residuals
    y_model_data = model(x_data, **model_params)
    residuals = y_data - y_model_data
      # Styling for main plot
    title_kwargs = {}
    ylabel_kwargs = {}
    xlabel_kwargs = {}
    legend_kwargs = {}
    
    # Get text color from palette
    text_color = str(palette.colours['text_primary']) if palette and 'text_primary' in palette.colours else 'black'
    
    if typography:
        title_kwargs = {'fontname': typography.title.font, 'fontsize': typography.title.size, 'color': text_color}
        ylabel_kwargs = {'fontname': typography.subtitle.font, 'fontsize': typography.subtitle.size, 'color': text_color}
        xlabel_kwargs = {'fontname': typography.subtitle.font, 'fontsize': typography.subtitle.size, 'color': text_color}
        legend_kwargs = {'fontsize': typography.body.size}
    else:
        title_kwargs = {'color': text_color}
        ylabel_kwargs = {'color': text_color}
        xlabel_kwargs = {'color': text_color}
    
    ax_main.set_title(title, fontweight='bold', pad=20, **title_kwargs)
    ax_main.set_ylabel(ylabel, **ylabel_kwargs)
    
    if not show_residuals:
        ax_main.set_xlabel(xlabel, **xlabel_kwargs)
    
    # Add grid and legend
    ax_main.grid(True, alpha=0.3, linestyle='--', color=str(palette.neutral_dark.lighten(20)) if palette else 'gray')
    neutral_color = str(palette.colours['neutral_light']) if palette else 'red'
    
    # Set border color to neutral_dark
    axes_border_color = str(palette.colours['neutral_dark']) if palette and 'neutral_dark' in palette.colours else 'black'
    ax_main.legend(frameon=True, fancybox=True, shadow=True, facecolor=neutral_color, 
                   edgecolor=axes_border_color, **legend_kwargs)
    # Set plot background color
    if palette:
        fig.patch.set_facecolor(neutral_color)
        ax_main.set_facecolor(neutral_color)
    
    # Style the main plot
    ax_main.spines['top'].set_linewidth(1.2)
    ax_main.spines['right'].set_linewidth(1.2)
    ax_main.spines['left'].set_linewidth(1.2)
    ax_main.spines['bottom'].set_linewidth(1.2)
    
    # Set spine colors to match legend edge
    for spine in ax_main.spines.values():
        spine.set_edgecolor(axes_border_color)
    
    # Set tick colors to match border and tick labels to match text
    ax_main.tick_params(axis='both', colors=axes_border_color, labelcolor=text_color)
    
    # Residuals plot
    if show_residuals and ax_residuals is not None:
        line_color = str(palette.colours['accent']) if palette else None

        ax_residuals.scatter(x_data, residuals, color=scatter_color, alpha=0.3, s=30,
                   edgecolors=str(palette.primary.darken(10)), linewidth=0.7,
                   rasterized=rasterize_points)
        ax_residuals.axhline(y=0, color=line_color, linestyle='-', linewidth=1.5, alpha=0.8)
        
        # Style residuals plot
        residuals_xlabel_kwargs = xlabel_kwargs if typography else {'color': text_color}
        residuals_ylabel_kwargs = ylabel_kwargs if typography else {'color': text_color}
        residuals_tick_kwargs = {'labelsize': typography.body.size-1} if typography else {}
        
        ax_residuals.set_xlabel(xlabel, **residuals_xlabel_kwargs)
        ax_residuals.set_ylabel('Residuals', **residuals_ylabel_kwargs)
        ax_residuals.grid(True, alpha=0.5, linestyle='--', color=str(palette.neutral_dark.lighten(20)) if palette else 'gray')
        
        # Style all spines for residuals plot
        ax_residuals.spines['top'].set_linewidth(1.2)
        ax_residuals.spines['right'].set_linewidth(1.2)
        ax_residuals.spines['left'].set_linewidth(1.2)
        ax_residuals.spines['bottom'].set_linewidth(1.2)
        
        # Set residuals spine colors to match
        for spine in ax_residuals.spines.values():
            spine.set_edgecolor(axes_border_color)
        
        # Set tick colors for residuals plot
        ax_residuals.tick_params(axis='both', colors=axes_border_color, labelcolor=text_color)
        
        # Make residuals plot smaller and align with main plot
        if typography:
            ax_residuals.tick_params(axis='both', **residuals_tick_kwargs, colors=axes_border_color, labelcolor=text_color)
        
        # Share x-axis with main plot
        ax_main.sharex(ax_residuals)
        ax_main.tick_params(labelbottom=False)  # Remove x-axis labels from main plot

        # set background color for residuals plot
        ax_residuals.set_facecolor(neutral_color)
    
    # Return appropriate axes
    if show_residuals and ax_residuals is not None:
        return fig, (ax_main, ax_residuals)
    else:
        return fig, ax_main