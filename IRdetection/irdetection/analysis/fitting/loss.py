from iminuit.cost import MaskedCostWithPulls


class CustomLoss(MaskedCostWithPulls):
    """
    Allows to use a custom function to compute the loss, making it ready for use with Minuit.
    
    """
    
    def __init__(self, cost, pulls=None, mask=None, params=None):
        super().__init__(cost, pulls=pulls, mask=mask)
        self.params = params if params is not None else {}
    
    def __call__(self, *args, **kwargs):
        # Call the parent method with additional parameters
        return super().__call__(*args, **kwargs) + sum(self.params.values())