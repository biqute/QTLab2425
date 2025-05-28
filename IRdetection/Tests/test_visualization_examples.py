"""
Test script for example visualization functionality.

This script demonstrates the example plot generation using a Coolors.co palette
and Palatino/Verdana typography. It tests the integration between the colors,
typography, plotting style, and example generation modules.
"""

import sys
import os
import matplotlib.pyplot as plt

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src/graphics'))

from graphics.typography import Typography, Font
from graphics.colours import Palette, coolors_palette
from graphics.examples import PlotExamples, quick_preview

# Define output directory
OUTPUT_DIR = "test_output_plots"

def ensure_output_directory():
    """Ensure the output directory exists."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

def create_test_typography():
    """
    Create a typography configuration using Palatino for titles and Verdana for body text.
    
    Returns
    -------
    Typography
        Configured typography object with Palatino/Verdana font combination.
    """
    typography = Typography(
        title=Font('Times New Roman', 18, family='serif'),        # Elegant serif for titles
        subtitle=Font('Palatino', 16, family='serif'),     # Consistent serif for subtitles
        body=Font('Verdana', 12, family='sans-serif'),          # Clean sans-serif for readability
        caption=Font('Verdana', 10, family='sans-serif')        # Smaller Verdana for captions
    )
    return typography


def create_test_palette_from_coolors():
    """
    Create a color palette from a Coolors.co link.
    
    Uses a professional, science-friendly color scheme suitable for
    academic and research publications.
    
    Returns
    -------
    Palette
        Color palette object created from Coolors.co color scheme.
    """
    # Professional color scheme: deep blue, teal, amber, coral, navy
    # coolors_link = "https://coolors.co/palette/264653-2a9d8f-e9c46a-f4a261-e76f51"
    
    # Extract colors from the link
    # colors = coolors_palette(coolors_link)

    palette = Palette("Science Professional", colours="5f0f40-9a031e-fb8b24-e36414-0f4c5c")
    
    # Create palette with meaningful color assignments
    # palette = Palette("Science Professional", {
    #     'primary': colors[0],      # Deep forest green - main data
    #     'secondary': colors[1],    # Teal - secondary data
    #     'accent': colors[2],       # Golden yellow - highlights
    #     'accent2': colors[3],      # Orange - secondary highlights
    #     'background': '#FFFFFF',   # Clean white background
    #     'background2': '#F8F9FA',  # Light gray secondary background
    #     'text_primary': '#2C3E50', # Dark blue-gray for text
    #     'text_secondary': '#7F8C8D', # Medium gray for secondary text
    #     'status_error': colors[4],  # Coral red for errors/warnings
    # })

    # palette = Palette("Science Professional", {
    #     'primary': '#264653',      # Deep blue - main data
    #     'secondary': '#2A9D8F',    # Teal - secondary data
    #     'accent': '#E9C46A',       # Amber - highlights
    #     'accent2': '#F4A261',      # Coral - secondary highlights
    #     'background': '#E6EBE0',   # Clean white background
    #     'background2': '#F8F9FA',  # Light gray secondary background
    #     'text_primary': '#264653', # Dark blue-gray for text
    #     'text_secondary': '#2A9D8F', # Medium teal for secondary text
    #     'status_error': '#E76F51',  # Coral red for errors/warnings
    # })
    
    return palette


def test_palette_visualization():
    """
    Test palette visualization by creating a color swatch display.
    """
    print("Testing palette visualization...")
    ensure_output_directory()
    
    palette = create_test_palette_from_coolors()
    
    try:
        # Create a simple palette visualization
        output_path = os.path.join(OUTPUT_DIR, "test_palette_visualization.png")
        palette.visualize_palette(title="Test Palette Visualization", save_path=output_path)

        print(f"    ✓ Palette visualization saved to: {output_path}")

    except Exception as e:
        print(f"    ✗ Error generating palette visualization: {e}")


def test_quick_preview():
    """
    Test the quick preview functionality with different style types.
    
    Generates quick preview plots for all available style types to demonstrate
    how the typography and palette combination looks across different contexts.
    """
    print("Testing quick preview functionality...")
    ensure_output_directory()
    
    typography = create_test_typography()
    palette = create_test_palette_from_coolors()
    
    style_types = ['minimal', 'publication', 'presentation', 'web']
    
    for style_type in style_types:
        try:
            output_path = os.path.join(OUTPUT_DIR, f"test_quick_preview_{style_type}.png")
            quick_preview(typography, palette, style_type, save_path=output_path)
            print(f"    ✓ {style_type} preview saved to: {output_path}")
            
        except Exception as e:
            print(f"    ✗ Error generating {style_type} preview: {e}")


def test_individual_examples():
    """
    Test individual example plot generation.
    
    Creates each type of example plot individually to verify that all
    plot types work correctly with the specified typography and palette.
    """
    print("\nTesting individual example plots...")
    ensure_output_directory()
    
    typography = create_test_typography()
    palette = create_test_palette_from_coolors()
    
    # Use publication style for individual tests
    examples = PlotExamples(typography, palette, 'publication')
    
    # Test each plot type
    plot_tests = [
        ('line_plot', examples.line_plot_example),
        ('scatter_plot', examples.scatter_plot_example),
        ('bar_chart', examples.bar_chart_example),
        ('heatmap', examples.heatmap_example),
        ('time_series', examples.time_series_example),
        ('subplots', examples.subplot_example)
    ]
    
    for plot_name, plot_function in plot_tests:
        try:
            fig = plot_function()
            output_path = os.path.join(OUTPUT_DIR, f"test_{plot_name}_example.png")
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"    ✓ {plot_name} example saved to: {output_path}")
            
        except Exception as e:
            print(f"    ✗ Error generating {plot_name} example: {e}")


def test_batch_generation():
    """
    Test batch generation of all example plots.
    """
    print("\nTesting batch generation...")
    ensure_output_directory()
    
    typography = create_test_typography()
    palette = create_test_palette_from_coolors()
    
    try:
        examples = PlotExamples(typography, palette, 'publication')
        examples.generate_all_examples(save_dir=OUTPUT_DIR)
        print(f"    ✓ Batch generation completed in: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"    ✗ Error in batch generation: {e}")


def run_all_tests():
    """
    Run all test functions and provide a summary.
    
    Executes all test functions in sequence and provides a summary
    of results and generated files.
    """
    print("=" * 60)
    print("TESTING EXAMPLE VISUALIZATION FUNCTIONALITY")
    print("=" * 60)
    print("Typography: Palatino (titles) + Verdana (body)")
    print("Palette: Science Professional (from Coolors.co)")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Ensure output directory exists
    ensure_output_directory()
    
    # Run all tests
    test_palette_visualization()
    test_quick_preview()
    test_individual_examples()
    test_batch_generation()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    # List all generated files in output directory
    if os.path.exists(OUTPUT_DIR):
        generated_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
        
        if generated_files:
            print(f"✓ Generated {len(generated_files)} test plots in '{OUTPUT_DIR}/':")
            for file in sorted(generated_files):
                print(f"  - {file}")
        else:
            print(f"✗ No test plots were generated in '{OUTPUT_DIR}/'")
    else:
        print(f"✗ Output directory '{OUTPUT_DIR}' was not created")
    
    print("\n" + "=" * 60)
    print(f"Test completed! Check the '{OUTPUT_DIR}/' directory to verify results.")
    print("=" * 60)


if __name__ == "__main__":
    # Change to the Tests directory if not already there
    if not os.path.basename(os.getcwd()) == 'Tests':
        os.chdir(os.path.dirname(__file__))

    
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
