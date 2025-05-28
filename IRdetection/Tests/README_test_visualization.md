# Visualization Examples Test

This test script demonstrates the example visualization functionality using a professional color palette from Coolors.co and Palatino/Verdana typography.

## Setup

Make sure you have the required dependencies installed:
```bash
pip install matplotlib seaborn numpy
```

## Running the Test

From the `Tests` directory, run:
```bash
python test_visualization_examples.py
```

## What it Tests

1. **Palette Creation**: Creates a color palette from a Coolors.co link
2. **Typography Setup**: Configures Palatino (titles) + Verdana (body) typography
3. **Quick Preview**: Generates preview plots for all style types
4. **Individual Examples**: Tests each plot type separately
5. **Batch Generation**: Creates all plots at once and saves them
6. **Palette Visualization**: Shows the color palette being used

## Output

The script generates several PNG files:
- `test_quick_preview_*.png`: Quick previews for each style type
- `test_*_example.png`: Individual plot type examples
- `test_palette_visualization.png`: Color palette visualization
- `test_output_plots/`: Directory with batch-generated plots

## Color Palette

Uses the "Science Professional" palette from Coolors.co:
- **Primary**: Deep forest green (#264653)
- **Secondary**: Teal (#2a9d8f) 
- **Accent**: Golden yellow (#e9c46a)
- **Accent2**: Orange (#f4a261)
- **Status Error**: Coral red (#e76f51)

## Typography

- **Titles**: Palatino 18pt (elegant serif)
- **Subtitles**: Palatino 16pt 
- **Body**: Verdana 12pt (clean sans-serif)
- **Captions**: Verdana 10pt

This combination provides excellent readability while maintaining a professional, academic appearance suitable for scientific publications.
