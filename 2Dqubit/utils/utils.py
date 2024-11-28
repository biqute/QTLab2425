import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
import os

class Data :


    def __init__ (self, x, y) :

        self.x = x
        self.y = y
        self.vect = np.column_stack((x, y))


    def save_txt (self, nome) :
        
        try :

            np.savetxt(f"../data/{nome}.txt", self.vect, fmt="%.18e", newline="\n", delimiter="\t")
            return True
        
        except Exception as e :

            print(f"Errore durante il salvataggio: {e}")
            return False
    
    
    #def read_txt (self, )
        
