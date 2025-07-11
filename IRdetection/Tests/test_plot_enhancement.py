#!/usr/bin/env python3
"""
Test script for the enhanced plot function with multiple plots support.
"""

import numpy as np
import sys
import os

# Add the IRdetection package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'IRdetection'))

from irdetection.graphics.plots import plot
from irdetection.graphics.palettes import QSciencePalette, QScienceTypography

def test_single_plot():
    """Test basic single plot functionality."""
    print("Testing single plot...")
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    fig, ax = plot(x, y, title="Single Plot Test", 
                   palette=QSciencePalette, typography=QScienceTypography,
                   xlabel="X values", ylabel="sin(x)", 
                   labels="sine wave", return_fig=True)
    print("✓ Single plot test passed")
    fig.show()  # Show the figure to verify single plot
    input("Press Enter to continue...")  # Pause for user to view the plot
    return fig

def test_multiple_plots_list():
    """Test multiple plots with lists of x and y."""
    print("Testing multiple plots with lists...")
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(x) * np.cos(x)
    
    fig, ax = plot([x, x, x], [y1, y2, y3], 
                   title="Multiple Plots Test", 
                   palette=QSciencePalette, typography=QScienceTypography,
                   xlabel="X values", ylabel="Trigonometric functions",
                   labels=["sin(x)", "cos(x)", "sin(x)cos(x)"], 
                   return_fig=True)
    print("✓ Multiple plots test passed")
    fig.show()  # Show the figure to verify multiple plots
    input("Press Enter to continue...")  # Pause for user to view the plot
    return fig

def test_multiple_plots_different_x():
    """Test multiple plots with different x ranges."""
    print("Testing multiple plots with different x ranges...")
    x1 = np.linspace(0, 5, 50)
    x2 = np.linspace(0, 10, 100)
    x3 = np.linspace(0, 15, 150)
    
    y1 = x1**2
    y2 = np.sin(x2)
    y3 = np.log(x3 + 1)
    
    fig, ax = plot([x1, x2, x3], [y1, y2, y3], 
                   title="Different X Ranges Test", 
                   palette=QSciencePalette, typography=QScienceTypography,
                   xlabel="X values", ylabel="Various functions",
                   labels=["x²", "sin(x)", "ln(x+1)"], 
                   return_fig=True)
    print("✓ Different x ranges test passed")
    fig.show()  # Show the figure to verify different x ranges
    input("Press Enter to continue...")  # Pause for user to view the plot
    return fig

def test_custom_colors():
    """Test multiple plots with custom colors."""
    print("Testing custom colors...")
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    custom_colors = ['red', '#00FF00']
    
    fig, ax = plot([x, x], [y1, y2], 
                   title="Custom Colors Test", 
                   palette=QSciencePalette, typography=QScienceTypography,
                   xlabel="X values", ylabel="Trigonometric functions",
                   labels=["sin(x)", "cos(x)"], 
                   colors=custom_colors,
                   return_fig=True)
    print("✓ Custom colors test passed")
    fig.show()  # Show the figure to verify custom colors
    input("Press Enter to continue...")  # Pause for user to view the plot
    return fig

def test_gradient_colors():
    """Test gradient color interpolation with palette colors."""
    print("Testing gradient colors...")
    x = np.linspace(0, 10, 100)
    
    # Create 8 different functions for a more comprehensive gradient demonstration
    y_functions = [
        np.sin(x + i * np.pi / 8) * np.exp(-x/10) for i in range(8)
    ]
    
    # Choose three nice colors for a smooth gradient: deep blue -> vibrant purple -> warm orange
    gradient_colors = [
        '#1f4e79',  # Deep blue
        '#8e44ad',  # Vibrant purple  
        '#e67e22'   # Warm orange
    ]
    
    # Create figure with subplot for color bar
    import matplotlib.pyplot as plt
    fig, (ax_main, ax_cbar) = plt.subplots(2, 1, figsize=(10, 8), 
                                          gridspec_kw={'height_ratios': [4, 0.3], 'hspace': 0.3})
    
    # Plot the main graph directly on ax_main (no labels to remove legend)
    fig_main, ax_plot = plot([x]*8, y_functions, 
                            title="Gradient Colors Test", 
                            palette=QSciencePalette, typography=QScienceTypography,
                            xlabel="X values", ylabel="Damped sine functions",
                            labels=None,  # Remove labels to hide legend
                            colors=gradient_colors,
                            gradient=True,
                            ax=ax_main,
                            return_fig=True)
    
    # Create color bar to visualize the gradient
    from matplotlib.colors import LinearSegmentedColormap
    
    # Create a custom colormap from our gradient colors
    cmap = LinearSegmentedColormap.from_list("custom_gradient", gradient_colors, N=256)
    
    # Create gradient bar
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax_cbar.imshow(gradient, aspect='auto', cmap=cmap, extent=[0, 7, 0, 1])
    ax_cbar.set_xlim(0, 7)
    ax_cbar.set_xlabel('Plot Number (shows color progression)', 
                       fontname=QScienceTypography.body.font if QScienceTypography else None,
                       fontsize=QScienceTypography.body.size if QScienceTypography else 12)
    ax_cbar.set_ylabel('Color\nGradient', 
                       fontname=QScienceTypography.body.font if QScienceTypography else None,
                       fontsize=QScienceTypography.body.size if QScienceTypography else 12)
    ax_cbar.set_yticks([])
    
    # Add text annotations for the key colors
    ax_cbar.text(0, 1.2, 'Deep Blue', ha='left', va='bottom', 
                fontsize=10, color=gradient_colors[0], weight='bold')
    ax_cbar.text(3.5, 1.2, 'Vibrant Purple', ha='center', va='bottom', 
                fontsize=10, color=gradient_colors[1], weight='bold')
    ax_cbar.text(7, 1.2, 'Warm Orange', ha='right', va='bottom', 
                fontsize=10, color=gradient_colors[2], weight='bold')
    
    # Style the color bar to match the palette
    if QSciencePalette:
        text_color = str(QSciencePalette.colours['text_primary'])
        axes_border_color = str(QSciencePalette.colours['neutral_dark'])
        neutral_color = str(QSciencePalette.colours['neutral_light'])
        
        ax_cbar.tick_params(axis='both', colors=axes_border_color, labelcolor=text_color)
        for spine in ax_cbar.spines.values():
            spine.set_edgecolor(axes_border_color)
            spine.set_linewidth(1.2)
        fig.patch.set_facecolor(neutral_color)
    
    print("✓ Gradient colors test passed")
    plt.show()  # Show the figure with both main plot and color bar
    input("Press Enter to continue...")  # Pause for user to view the plot
    return fig

def test_auto_color_cycling():
    """Test automatic color cycling when more plots than available colors."""
    print("Testing automatic color cycling...")
    x = np.linspace(0, 10, 50)
    
    # Create 6 different functions (more than typical palette colors)
    y_functions = [
        np.sin(x),
        np.cos(x), 
        np.sin(2*x),
        np.cos(2*x),
        np.sin(x/2),
        np.cos(x/2)
    ]
    
    labels = ["sin(x)", "cos(x)", "sin(2x)", "cos(2x)", "sin(x/2)", "cos(x/2)"]
    
    fig, ax = plot([x]*6, y_functions, 
                   title="Color Cycling Test", 
                   palette=QSciencePalette, typography=QScienceTypography,
                   xlabel="X values", ylabel="Trigonometric functions",
                   labels=labels, 
                   return_fig=True)
    print("✓ Color cycling test passed")
    fig.show()  # Show the figure to verify color cycling
    input("Press Enter to continue...")  # Pause for user to view the plot
    return fig

def test_backward_compatibility():
    """Test that the function still works with old single-plot syntax."""
    print("Testing backward compatibility...")
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # Test old way of calling (should still work)
    fig, ax = plot(x, y, title="Backward Compatibility Test", 
                   palette=QSciencePalette, typography=QScienceTypography,
                   return_fig=True)
    print("✓ Backward compatibility test passed")
    fig.show()  # Show the figure to verify backward compatibility
    input("Press Enter to continue...")  # Pause for user to view the plot
    return fig

if __name__ == "__main__":
    print("Testing enhanced plot function...")
    print("=" * 50)
    
    try:
        # Run all tests
        test_single_plot()
        test_multiple_plots_list()
        test_multiple_plots_different_x()
        test_custom_colors()
        test_gradient_colors()
        test_auto_color_cycling()
        test_backward_compatibility()
        
        print("=" * 50)
        print("✅ All tests passed successfully!")
        print("The enhanced plot function is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
