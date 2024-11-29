import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
import os

class Data :


    def __init__ (self, x, y) :

        if len (x) != len (y) :
            raise ValueError("I vettori x e y devono avere la stessa lunghezza.")

        self.x = np.array(x)
        self.y = np.array(y)
        self.vect = np.column_stack((self.x, self.y))


    def save_txt (self, nome, commento="colonne x e y") :
        
        try :

            np.savetxt(f"../data/{nome}.txt", self.vect, fmt="%.18g", newline="\n", delimiter="\t", header=commento)
            return True
        
        except Exception as e :

            print(f"Errore durante il salvataggio: {e}")
            return False
        

    def plot (self, x= None, y= None, Title="Spectrum", Nome_x="Frequency [Hz]", Nome_y="Amplitude [dBm]") :

        if x is None:
            x = self.x

        if y is None:
            y = self.y

        fig, ax = plt.subplots ( nrows=1, ncols=1)
        plt.title(f"{Title}")
        plt.xlabel(f"{Nome_x}")
        plt.ylabel(f"{Nome_y}")
        plt.plot(x, y)
        plt.show()
    
    
    @staticmethod
    def read_txt (nome) :

        try:

            data = np.loadtxt(f"../data/{nome}.txt", delimiter="\t")
            x = data[:, 0]
            y = data[:, 1]
            return x, y
        
        except Exception as e :

            print(f"Errore durante la lettura: {e}")
            return False
        
