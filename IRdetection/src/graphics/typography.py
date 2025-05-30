class Font:
    """
    Font class for managing font styles and sizes.

    Attributes:
        family (str): The font family to use.
        size (int): The font size to use.

    Methods:
        up(amount: int): Increases the font size by the specified amount. Default is 1.
        down(amount: int): Decreases the font size by the specified amount. Default is 1.
    """

    def __init__(self, font: str, size: int, family: str = 'sans-serif'):
        """
        Initializes the Font object with a specific font family and size.
        
        Parameters
        ----------
        font : str
            The font family name (e.g., 'Arial', 'Times New Roman').
        size : int
            The size of the font in points.
        family : str, optional
            The font family to use. Default is 'sans-serif'.
        """
        self.font = font
        self.family = family
        self.size = size

    def __repr__(self):
        return f"Font(font='{self.font}', family='{self.family}', size={self.size})"

    def up(self, amount: int = 1):
        """
        Increases the font size by the specified amount.

        Args:
            amount (int): The amount to increase the font size by. Default is 1.
        """
        self.size += amount

    def down(self, amount: int = 1):
        """
        Decreases the font size by the specified amount.

        Args:
            amount (int): The amount to decrease the font size by. Default is 1.
        """
        self.size -= amount

class Typography:
    """
    Typography class for managing font styles and sizes for a graphic style.

    Attributes:
        title (Font): Font for titles.
        subtitle (Font): Font for subtitles.
        body (Font): Font for body text.
        caption (Font): Font for captions.
    """
    def __init__(self, title: Font, subtitle: Font, body: Font, caption: Font):
        self.title = title
        self.subtitle = subtitle
        self.body = body
        self.caption = caption

    def __repr__(self):
        return (f"Typography(title={self.title}, subtitle={self.subtitle}, "
                f"body={self.body}, caption={self.caption})")