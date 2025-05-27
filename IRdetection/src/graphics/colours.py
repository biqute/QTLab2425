"""
This module contains colors and utility functions for color manipulation.
"""

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import seaborn as sns
import re


class Colour:
    """
    A class representing a colour.

    Parameters
    ----------
    name : str
        The name of the colour.
    colour : str, list, tuple
        The colour in hex format or as an RGB/RGBA tuple.

    Attributes
    -------
    name : str
        The name of the colour.
    rgb : tuple
        The RGB representation of the colour.
    hex : str
        The hex representation of the colour.
    rgba : tuple
        The RGBA representation of the colour.
    alpha : float
        The alpha value of the colour.
    R : int
        The red component of the colour.
    G : int
        The green component of the colour.
    B : int
        The blue component of the colour.

    Methods
    -------
    __str__()
        Returns the hex string representation of the colour.
    __eq__(other)
        Checks if two Colour objects are equal.
    darken(factor)
        Darkens the colour by a given factor (percentage).
    lighten(factor)
        Lightens the colour by a given factor (percentage).
      
    """

    def __init__(self, colour):
        if isinstance(colour, str):
            self.hex = colour
            self.rgb = self._hex_to_rgb(colour)
            self.rgba = self._hex_to_rgba(colour)
        elif isinstance(colour, (list, tuple)) and len(colour) in [3, 4]:
            self.rgb = tuple(colour[:3])
            self.rgba = tuple(colour)
            self.hex = self._rgb_to_hex(self.rgb)
        else:
            raise ValueError("Invalid colour format. Use hex string or RGB/RGBA tuple.")
        
        self.alpha = self.rgba[3] if len(colour) == 4 else 1.0
        self.R, self.G, self.B = self.rgb
    
    def __str__(self):
        """
        Returns the hex string representation of the colour.
        """
        return self.hex

    def __eq__(self, other :'Colour'):
        """
        Checks if two Colour objects are equal.
        """
        return self.rgba == other.rgba
    

    def darken(self, factor):
        """
        Darkens the colour by a given factor (percentage).
        """
        factor = max(0, min(100, factor))
        dR = int(self.R * (1 - factor / 100))
        dG = int(self.G * (1 - factor / 100))
        dB = int(self.B * (1 - factor / 100))
        dhex = self._rgb_to_hex((dR, dG, dB))
        return Colour(dhex)

    def lighten(self, factor):
        """
        Lightens the colour by a given factor (percentage).
        """
        factor = max(0, min(100, factor))
        lR = int(self.R + (255 - self.R) * (factor / 100))
        lG = int(self.G + (255 - self.G) * (factor / 100))
        lB = int(self.B + (255 - self.B) * (factor / 100))
        lhex = self._rgb_to_hex((lR, lG, lB))
        return Colour(lhex)
    
    def _hex_to_rgb(self, hex_color):
        """
        Converts a hex color string to an RGB tuple.
        """
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    
    def _rgb_to_hex(self, rgb_color):
        """
        Converts an RGB tuple to a hex color string.
        """
        return '#' + ''.join(f'{int(c):02x}' for c in rgb_color).upper()
    
    def _hex_to_rgba(self, hex_color):
        """
        Converts a hex color string to an RGBA tuple.
        """
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4)) + (1.0,)
        elif len(hex_color) == 8:
            return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4, 6))
        else:
            raise ValueError("Invalid hex color format. Use #RRGGBB or #RRGGBBAA.")


COLOURS = {
    "red": Colour("#FF0000"),
    "green": Colour("#00FF00"),
    "blue": Colour("#0000FF"),
    "yellow": Colour("#FFFF00"),
    "cyan": Colour("#00FFFF"),
    "magenta": Colour("#FF00FF"),
    "black": Colour("#000000"),
    "pastel_black": Colour("#1C1C1C"),
    "pastel_white": Colour("#F5F5F5"),
    "dark_gray": Colour("#3E3E3E"),
    "white": Colour("#FFFFFF"),
    "gray": Colour("#808080"),
    "orange": Colour("#FFA500"),
    "purple": Colour("#800080"),
    "pink": Colour("#FFC0CB"),
    'crimson': Colour("#DC143C"),
    "gold": Colour("#FFD700"),
    "navy": Colour("#000080"),
    'webgreen': Colour("#00FF7F"),
    "webblue": Colour("#1E90FF"),
    "webred": Colour("#FF4500"),
    "webpurple": Colour("#800080"),
    "webyellow": Colour("#FFFF00"),
}

class Palette:
    """
    A class representing a color palette.

    Parameters
    ----------
    name : str
        Name of the palette.
    colours : list
        List of colours in the palette or dictionary with elements as keys and colours as values.
        elements are:
        - primary
        - secondary
        - accent
        - accent2 : default is accent darkened by 20%
        - background
        - background2
        - text_primary : default is black
        - text_secondary : default is dark_gray
        - text_accent : default is crimson
        - neutral_dark : default is pastel_black
        - neutral_light : default is pastel_white
        - status_success : default is webgreen
        - status_warning : default is webyellow
        - status_error : default is webred
        - status_info : default is webblue

        If you pass a list it can contain only 5 elements which will be used as:
        - primary
        - secondary
        - accent
        - background
        - background2
        
        To set other elements you can use Palette.<element> = <color>.
        If you want to add more elements you can use the add_element method.

    Methods
    -------
    save_palette(filename)
        Saves the palette to a json file.
    load_palette(filename)
        Loads the palette from a json file.
    visualize_palette()
        Shows the palette in a matplotlib figure.
    
    """
    def __init__(self, name, colours):
        self.name = name
        # Set default colours
        self.colours = {
            "text_primary": COLOURS["black"],
            "text_secondary": COLOURS["dark_gray"],
            "text_accent": COLOURS["crimson"],
            "neutral_dark": COLOURS["pastel_black"],
            "neutral_light": COLOURS["pastel_white"],
            "status_success": COLOURS["webgreen"],
            "status_warning": COLOURS["webyellow"],
            "status_error": COLOURS["webred"],
            "status_info": COLOURS["webblue"]
        }
        if isinstance(colours, dict):
            self.colours.update(colours)
        elif isinstance(colours, list) and len(colours) == 5:
            self.colours.update({
                "primary": colours[0],
                "secondary": colours[1],
                "accent": colours[2],
                "background": colours[3],
                "background2": colours[4],
            })
        else:
            raise ValueError("Invalid colours input.")
        # set accent2 if not set
        if "accent2" not in self.colours:
            self.colours["accent2"] = self.colours["accent"].darken(20)

    def save_palette(self, filename):
        """
        Saves the palette to a json file.

        Parameters
        ----------
        filename : str
            The name of the file to save the palette to.
        """
        import json
        with open(filename, 'w') as f:
            json.dump(self.colours, f)

    def load_palette(self, filename):
        """
        Loads the palette from a json file.

        Parameters
        ----------
        filename : str
            The name of the file to load the palette from.
        """
        import json
        with open(filename, 'r') as f:
            self.colours = json.load(f)

    def visualize_palette(self):
        """
        Shows the palette in a matplotlib figure.
        """
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define the order of importance for displaying colors
        color_order = [
            "primary", "secondary", "accent", "accent2",
            "background", "background2", 
            "text_primary", "text_secondary", "text_accent",
            "neutral_dark", "neutral_light",
            "status_success", "status_warning", "status_error", "status_info"
        ]
        
        # Create ordered list of colors, putting ordered ones first, then any extras
        ordered_colors = []
        for key in color_order:
            if key in self.colours:
                ordered_colors.append((key, self.colours[key]))
        
        # Add any remaining colors not in the predefined order
        for key, color in self.colours.items():
            if key not in color_order:
                ordered_colors.append((key, color))
        
        num_colors = len(ordered_colors)
        cols = min(4, num_colors)
        rows = (num_colors + cols - 1) // cols
        
        for i, (name, colour) in enumerate(ordered_colors):
            row = i // cols
            col = i % cols
            
            # Get hex value properly
            hex_val = colour.hex if hasattr(colour, 'hex') else str(colour)
            
            # Create rectangle
            rect = mpatches.Rectangle((col * 2, (rows - row - 1) * 2), 1.8, 1.8, 
                                    facecolor=hex_val, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            # Add color name
            ax.text(col * 2 + 0.9, (rows - row - 1) * 2 + 1.4, name.replace('_', ' ').title(), 
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   color='white' if sum(colour.rgb) < 384 else 'black')
            
            # Add hex value
            ax.text(col * 2 + 0.9, (rows - row - 1) * 2 + 0.4, hex_val, 
                   ha='center', va='center', fontsize=8,
                   color='white' if sum(colour.rgb) < 384 else 'black')
        
        ax.set_xlim(-0.1, cols * 2)
        ax.set_ylim(-0.1, rows * 2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"Color Palette: {self.name}", fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.show()


def coolors_palette(link):
    """
    Generates a color palette from a Coolors.co link.
    Links are like: https://coolors.co/palette/264653-2a9d8f-e9c46a-f4a261-e76f51
    where the colors are separated by a dash.

    Parameters
    ----------
    link : str
        The Coolors.co link to the palette.

    Returns
    -------
    list
        A list of Colour objects representing the palette.
    """
    pattern = r'(?<=palette/)([0-9a-fA-F]{6}(?:-[0-9a-fA-F]{6})*)'
    match = re.search(pattern, link)
    if match:
        hex_colors = match.group(0).split('-')
        return [Colour('#' + hex_color) for hex_color in hex_colors]
    else:
        raise ValueError("Invalid Coolors.co link format.")
    

if __name__ == "__main__":
    colours = coolors_palette("https://coolors.co/palette/264653-2a9d8f-e9c46a-f4a261-e76f51")
    print(colours)
    my_palette = Palette("My Coolors Palette", colours)
    my_palette.visualize_palette()
