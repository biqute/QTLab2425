import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from irdetection.graphics.colours import Palette
from irdetection.graphics.typography import Typography


def _get_plot_colors(palette: Palette, n_colors: int = None):
    """
    Get a list of colors suitable for plotting from a palette.
    
    Parameters
    ----------
    palette : Palette
        The color palette to extract colors from.
    n_colors : int, optional
        Number of colors needed. If None, returns the default cycling colors.
        
    Returns
    -------
    list
        List of color strings suitable for plotting.
    """
    if palette is None:
        # Default matplotlib colors if no palette provided
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        return default_colors[:n_colors] if n_colors else default_colors
    
    # Define the preferred order of colors for plotting from a palette
    color_priority = ['primary', 'accent', 'secondary', 'accent2', 'status_info', 
                     'status_success', 'status_warning', 'status_error']
    
    plot_colors = []
    for color_name in color_priority:
        if color_name in palette.colours:
            plot_colors.append(str(palette.colours[color_name]))
    
    # If we still need more colors and haven't reached the target, cycle through what we have
    if n_colors and len(plot_colors) < n_colors:
        plot_colors = (plot_colors * ((n_colors // len(plot_colors)) + 1))[:n_colors]
    
    return plot_colors if plot_colors else ['blue']  # Fallback


def _interpolate_colors(colors, n_colors):
    """
    Interpolate between a list of colors to generate a gradient with n_colors.
    
    Parameters
    ----------
    colors : list
        List of color strings to interpolate between.
    n_colors : int
        Number of colors to generate in the gradient.
        
    Returns
    -------
    list
        List of interpolated color strings.
    """
    if len(colors) < 2:
        # If only one color provided, return that color repeated
        return [colors[0]] * n_colors if colors else ['blue'] * n_colors
    
    if n_colors <= len(colors):
        # If we need fewer colors than provided, just take a subset
        return colors[:n_colors]
    
    # Import matplotlib.colors for color interpolation
    from matplotlib.colors import to_rgb, to_hex
    import numpy as np
    
    # Convert colors to RGB
    rgb_colors = [to_rgb(color) for color in colors]
    
    # Create interpolation points
    original_positions = np.linspace(0, 1, len(rgb_colors))
    target_positions = np.linspace(0, 1, n_colors)
    
    # Interpolate each RGB channel
    interpolated_colors = []
    for pos in target_positions:
        # Find the two colors to interpolate between
        idx = np.searchsorted(original_positions, pos)
        
        if idx == 0:
            interpolated_colors.append(rgb_colors[0])
        elif idx >= len(rgb_colors):
            interpolated_colors.append(rgb_colors[-1])
        else:
            # Linear interpolation between two colors
            t = (pos - original_positions[idx-1]) / (original_positions[idx] - original_positions[idx-1])
            rgb1 = np.array(rgb_colors[idx-1])
            rgb2 = np.array(rgb_colors[idx])
            interpolated_rgb = rgb1 + t * (rgb2 - rgb1)
            interpolated_colors.append(tuple(interpolated_rgb))
    
    # Convert back to hex strings
    return [to_hex(rgb) for rgb in interpolated_colors]


def plot_fit(data, model, model_params, title: str = 'Model Fit', palette: Palette = None, typography: Typography = None, 
             xlabel: str = 'X-axis', ylabel: str = 'Y-axis', data_label: str = 'Data', model_label: str = 'Model', show_residuals: bool = True,
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
    data_label : str, optional
        Label for the data points in the legend. Default is 'Data'.
    model_label : str, optional
        Label for the model line in the legend. Default is 'Model'.
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
        (fig, axes) where axes is [ax_main] or [ax_main, ax_residuals] depending on show_residuals.    
        
    """
    # Create or subdivide axes
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
                   label=data_label, zorder=2, edgecolors=str(palette.primary.darken(10)), linewidth=0.8, rasterized=rasterize_points, **kwargs)
    
    # Generate smooth model curve
    x_fit = np.linspace(x_data.min(), x_data.max(), 300)
    y_fit = model(x_fit, **model_params)

    line_color = str(palette.colours['accent']) if palette else None
    ax_main.plot(x_fit, y_fit, color=line_color, linewidth=2, 
                label=model_label, zorder=3, alpha=0.9)
    
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
    

def plot(x, y, title: str = 'Plot', palette: Palette = None, typography: Typography = None, 
         xlabel: str = 'X-axis', ylabel: str = 'Y-axis', figsize: tuple = (10, 6), ax=None, grid: bool = True, 
         return_fig: bool = False, labels=None, colors=None, gradient: bool = False, **kwargs):
    """
    Plot using matplotlib with customized aesthetics. Supports both single and multiple plots.

    Parameters
    ----------
    x : array-like or list of array-like
        X data for the plot(s). Can be a single array or list of arrays for multiple plots.
    y : array-like or list of array-like
        Y data for the plot(s). Can be a single array or list of arrays for multiple plots.
    title : str, optional
        Title of the plot.
    palette : Palette, optional
        Color palette for the plot. If provided, colors will be automatically cycled from the palette.
    typography : Typography, optional
        Typography settings for the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    figsize : tuple, optional
        Figure size in inches.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    grid : bool, optional
        Whether to show grid lines. Default is True.
    return_fig : bool, optional
        Whether to return the figure object. Default is False.
    labels : str, list of str, optional
        Legend labels for the plot(s). Can be a single string or list of strings.
    colors : str, list of str, optional
        Colors for the plot(s). Can be a single color or list of colors. If not provided,
        colors will be automatically cycled from the palette.
    gradient : bool, optional
        If True, interpolates between the colors in the 'colors' list to create a gradient
        across all plots. If False, colors are used as discrete values. Default is False.
    **kwargs : keyword arguments
        Additional keyword arguments for the plot.

    Returns
    -------
    matplotlib.figure.Figure or tuple
        The figure object (and axes if return_fig=True) for the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Normalize inputs to lists for consistent handling
    def _normalize_to_list(data):
        """Convert single items or lists to lists, handling nested structure properly."""
        if isinstance(data, (list, tuple)):
            # Check if it's a list of arrays/lists (multiple plots) or a single array
            if len(data) > 0 and hasattr(data[0], '__len__') and not isinstance(data[0], str):
                # If first element has length and isn't a string, assume it's multiple plots
                return list(data)
            else:
                # Single plot data
                return [data]
        else:
            # Single array
            return [data]
    
    # Handle both single and multiple plot cases
    x_data = _normalize_to_list(x)
    y_data = _normalize_to_list(y)
    
    # Ensure x and y have the same number of datasets
    if len(x_data) == 1 and len(y_data) > 1:
        # Broadcast single x to match multiple y
        x_data = x_data * len(y_data)
    elif len(y_data) == 1 and len(x_data) > 1:
        # Broadcast single y to match multiple x
        y_data = y_data * len(x_data)
    elif len(x_data) != len(y_data):
        raise ValueError(f"Number of x datasets ({len(x_data)}) must match number of y datasets ({len(y_data)})")
    
    n_plots = len(y_data)
    
    # Handle labels
    if labels is None:
        plot_labels = [None] * n_plots
    elif isinstance(labels, str):
        plot_labels = [labels] if n_plots == 1 else [f"{labels} {i+1}" for i in range(n_plots)]
    elif isinstance(labels, (list, tuple)):
        plot_labels = list(labels)
        # Pad with None if not enough labels provided
        while len(plot_labels) < n_plots:
            plot_labels.append(None)
    else:
        plot_labels = [None] * n_plots
    
    # Handle colors
    if colors is None:
        plot_colors = _get_plot_colors(palette, n_plots)
    elif isinstance(colors, str):
        plot_colors = [colors] * n_plots
    elif isinstance(colors, (list, tuple)):
        if gradient and len(colors) >= 2:
            # Use gradient interpolation between the provided colors
            plot_colors = _interpolate_colors(list(colors), n_plots)
        else:
            # Use colors as discrete values
            plot_colors = list(colors)
            # If not enough colors provided, cycle through available ones
            if len(plot_colors) < n_plots:
                color_cycle = cycle(plot_colors)
                plot_colors = [next(color_cycle) for _ in range(n_plots)]
    else:
        plot_colors = _get_plot_colors(palette, n_plots)

    # Set title and labels
    text_color = str(palette.colours['text_primary']) if palette and 'text_primary' in palette.colours else 'black'
    title_kwargs = {'fontname': typography.title.font, 'fontsize': typography.title.size, 'color': text_color} if typography else {'color': text_color}
    ylabel_kwargs = {'fontname': typography.subtitle.font, 'fontsize': typography.subtitle.size, 'color': text_color} if typography else {'color': text_color}
    xlabel_kwargs = {'fontname': typography.subtitle.font, 'fontsize': typography.subtitle.size, 'color': text_color} if typography else {'color': text_color}
    ax.set_title(title, fontweight='bold', pad=20, **title_kwargs)
    ax.set_ylabel(ylabel, **ylabel_kwargs)
    ax.set_xlabel(xlabel, **xlabel_kwargs)
    
    grid_color = str(palette.neutral_dark.lighten(20)) if palette and hasattr(palette, 'neutral_dark') else 'gray'
    ax.grid(grid, alpha=0.3, linestyle='--', color=grid_color)

    # Set background color
    neutral_color = str(palette.colours['neutral_light']) if palette and 'neutral_light' in palette.colours else 'white'
    ax.set_facecolor(neutral_color)
    fig.patch.set_facecolor(neutral_color)
    
    # Set border color
    axes_border_color = str(palette.colours['neutral_dark']) if palette and 'neutral_dark' in palette.colours else 'black'
    ax.spines['top'].set_color(axes_border_color)
    ax.spines['right'].set_color(axes_border_color)
    ax.spines['left'].set_color(axes_border_color)
    ax.spines['bottom'].set_color(axes_border_color)
    ax.spines['top'].set_linewidth(1.2)
    ax.spines['right'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Set tick colors
    ax.tick_params(axis='both', colors=axes_border_color, labelcolor=text_color)
    
    # Plot all datasets
    lines = []
    for i in range(n_plots):
        # Extract kwargs for this specific plot, avoiding conflicts
        plot_kwargs = kwargs.copy()
        if 'color' in plot_kwargs:
            plot_kwargs.pop('color')  # Use our color management instead
        if 'label' in plot_kwargs:
            plot_kwargs.pop('label')  # Use our label management instead
            
        line = ax.plot(x_data[i], y_data[i], color=plot_colors[i], linewidth=2, 
                      label=plot_labels[i], **plot_kwargs)
        lines.extend(line)
    
    # Only show legend if there are actual labels to show
    if any(label is not None for label in plot_labels):
        legend_kwargs = {'fontsize': typography.body.size} if typography else {}
        legend_kwargs.update({
            'frameon': True, 
            'fancybox': True, 
            'shadow': True, 
            'facecolor': neutral_color,
            'edgecolor': axes_border_color
        })
        ax.legend(**legend_kwargs)
    
    # Use tight_layout without rect to avoid spacing issues
    fig.tight_layout()
    
    # Return the figure for display
    if return_fig:
        return fig, ax
    else:
        plt.show()

