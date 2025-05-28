import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from typography import Typography
from colours import Colour, Palette

_TITLE_FUNCS_PATCHED = False

def _patch_title_functions(typography: Typography):
    global _TITLE_FUNCS_PATCHED
    if _TITLE_FUNCS_PATCHED:
        return
    from matplotlib import axes, figure
    _old_set_title = axes.Axes.set_title
    def _set_title(self, label, *args, **kwargs):
        kwargs.setdefault('fontname', typography.title.font)
        kwargs.setdefault('fontsize', typography.title.size)
        return _old_set_title(self, label, *args, **kwargs)
    axes.Axes.set_title = _set_title
    _old_suptitle = figure.Figure.suptitle
    def _suptitle(self, t, *args, **kwargs):
        kwargs.setdefault('fontname', typography.title.font)
        kwargs.setdefault('fontsize', typography.title.size + 2)
        return _old_suptitle(self, t, *args, **kwargs)
    figure.Figure.suptitle = _suptitle
    _TITLE_FUNCS_PATCHED = True

def set_plotting_style(typography: Typography, palette: Palette, style_type: str = 'publication'):
    _patch_title_functions(typography)
    color_cycle = _create_color_cycle(palette)
    rcParams.update({
        'font.family': typography.body.family,
        'font.sans-serif': [typography.body.font],
        'font.serif': [typography.title.font],
        'font.size': typography.body.size,
        'font.weight': 'normal',
        'axes.titlesize': typography.title.size,
        'axes.titleweight': 'bold',
        'axes.titlepad': 20,
        'axes.labelsize': typography.subtitle.size,
        'axes.labelweight': 'normal',
        'axes.labelpad': 8,
        'xtick.labelsize': typography.body.size,
        'ytick.labelsize': typography.body.size,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.minor.width': 0.8,
        'ytick.minor.width': 0.8,
        'legend.fontsize': typography.caption.size,
        'legend.title_fontsize': typography.body.size,
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': False,
        'legend.framealpha': 0.9,
        'legend.edgecolor': str(palette.colours.get('neutral_dark', '#000000')),
        'legend.facecolor': str(palette.colours.get('background', '#FFFFFF')),
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'figure.facecolor': str(palette.colours.get('background', '#FFFFFF')),
        'figure.edgecolor': 'none',
        'figure.autolayout': True,
        'axes.facecolor': str(palette.colours.get('background', '#FFFFFF')),
        'axes.edgecolor': str(palette.colours.get('text_secondary', '#333333')),
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'axes.axisbelow': True,
        'axes.prop_cycle': plt.cycler('color', color_cycle),
        'grid.color': str(palette.colours.get('neutral_light', '#E0E0E0')),
        'grid.linestyle': '-',
        'grid.linewidth': 0.8,
        'grid.alpha': 0.6,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'lines.markeredgewidth': 1.5,
        'lines.solid_capstyle': 'round',
        'patch.edgecolor': str(palette.colours.get('text_secondary', '#333333')),
        'patch.linewidth': 1.0,
        'text.color': str(palette.colours.get('text_primary', '#000000')),
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,
        'savefig.facecolor': str(palette.colours.get('background', '#FFFFFF')),
        'savefig.edgecolor': 'none',
    })
    _apply_style_modifications(style_type, palette, typography)
    sns.set_theme(
        style='whitegrid' if style_type != 'dark' else 'darkgrid',
        palette=color_cycle,
        font=typography.body.family,
        font_scale=1.0,
        color_codes=True,
        rc={
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.bottom': True,
            'xtick.top': False,
            'ytick.left': True,
            'ytick.right': False,
        }
    )
    sns.set_style('ticks', {'grid.linestyle': '-'})

def _create_color_cycle(palette: Palette) -> list:
    cycle_colors = []
    priority_colors = ['primary', 'secondary', 'accent', 'accent2']
    for color_name in priority_colors:
        if color_name in palette.colours:
            cycle_colors.append(str(palette.colours[color_name]))
    status_colors = ['status_info', 'status_success', 'status_warning', 'status_error']
    for color_name in status_colors:
        if color_name in palette.colours and len(cycle_colors) < 8:
            cycle_colors.append(str(palette.colours[color_name]))
    while len(cycle_colors) < 6:
        base_color = palette.colours.get('primary', Colour('#1f77b4'))
        if isinstance(base_color, Colour):
            variation_factor = 20 * (len(cycle_colors) - 3)
            if variation_factor > 0:
                new_color = base_color.lighten(variation_factor)
            else:
                new_color = base_color.darken(abs(variation_factor))
            cycle_colors.append(str(new_color))
        else:
            fallback_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            cycle_colors.append(fallback_colors[len(cycle_colors) % len(fallback_colors)])
    return cycle_colors

def _apply_style_modifications(style_type: str, palette: Palette, typography: Typography):
    if style_type == 'presentation':
        rcParams.update({
            'font.size': typography.body.size + 2,
            'axes.titlesize': typography.title.size + 4,
            'axes.labelsize': typography.subtitle.size + 2,
            'lines.linewidth': 3.0,
            'lines.markersize': 10,
            'xtick.major.size': 8,
            'ytick.major.size': 8,
            'grid.linewidth': 1.0,
            'axes.linewidth': 1.5,
        })
    elif style_type == 'notebook':
        rcParams.update({
            'figure.figsize': (12, 8),
            'lines.linewidth': 2.0,
            'grid.alpha': 0.4,
            'axes.grid': True,
        })
    elif style_type == 'dark':
        dark_bg = str(palette.colours.get('neutral_dark', '#2E2E2E'))
        light_text = str(palette.colours.get('neutral_light', '#FFFFFF'))
        rcParams.update({
            'figure.facecolor': dark_bg,
            'axes.facecolor': dark_bg,
            'text.color': light_text,
            'axes.edgecolor': light_text,
            'xtick.color': light_text,
            'ytick.color': light_text,
            'axes.labelcolor': light_text,
            'grid.color': str(palette.colours.get('text_secondary', '#666666')),
            'savefig.facecolor': dark_bg,
        })
    elif style_type == 'publication':
        rcParams.update({
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 1.5,
            'font.size': typography.body.size,
        })


#prova