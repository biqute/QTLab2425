from irdetection.graphics.colours import Palette
from irdetection.graphics.typography import Typography, Font

QSciencePalette = Palette("Quantum Science", {
    'primary': "#046A3A",      # Deep ocean blue - sophisticated and trustworthy
    'secondary': '#1B263B',    # Midnight navy - elegant depth
    'accent': "#E65039",       # Vibrant coral - high-impact highlights
    'accent2': '#FFD23F',      # Golden yellow - energy and precision
    'background': '#F8F9FB',   # Ultra-clean off-white - pristine lab feel
    'background2': '#E8EDF3',  # Subtle blue-gray - sophisticated secondary
    'text_primary': '#1A1B23', # Rich charcoal - maximum readability
    'text_secondary': '#4A5568', # Cool gray - perfect for secondary text
    'status_error': '#E53E3E',  # Crisp red - clear error indication
    'neutral_light': "#F3F3F3",  # Cool light gray - subtle grid
    'neutral_dark': "#2D3748",   # Professional dark gray - strong borders
})

QScienceTypography = Typography(
    title=Font('Palatino Linotype', 20, family='serif'),   
    subtitle=Font('Verdana', 16, family='sans-serif'),
    body=Font('Verdana', 13, family='sans-serif'),    
    caption=Font('Verdana', 11, family='sans-serif')  
)