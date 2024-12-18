import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from matplotlib.gridspec import GridSpec
import scipy.optimize as spopt
from scipy import stats


def read_data(file_path):
    '''
    funzione per leggere i dati
    '''
    data = np.loadtxt(file_path, delimiter=',')  
    f = data[:, 0]     # frequenza
    x = data[:, 1]     # parte reale (I)
    y = data[:, 2]     # parte immaginaria (Q)
    return f, x, y


def filter_data(I, Q, I_range, Q_range):
    """
    Filtra i dati mantenendo solo le coppie di I e Q che rientrano
    nei range specificati.

    Parameters:
        I (array): Vettore delle componenti I.
        Q (array): Vettore delle componenti Q.
        I_range (tuple): Intervallo per I (min, max).
        Q_range (tuple): Intervallo per Q (min, max).

    Returns:
        I_filtered, Q_filtered: Vettori filtrati.
    """
    mask = (I >= I_range[0]) & (I <= I_range[1]) & (Q >= Q_range[0]) & (Q <= Q_range[1])
    return I[mask], Q[mask]




# -------------------------------- FIT DELLA RISONANZA ---------------------------------

def skewed_lorentzian(f, A, B, C, D, QL, fr):
    return A + B * (f - fr) + (C + D * (f - fr)) / (1 + 4 * QL**2 * ((f - fr) / fr)**2)

def full_fit (f, a, a0, a1, a2, a3, L, C, fr, phi, fmin) :
    return ((a + a0*f + a1*f**2 + a2*f**3) + a3*np.abs(1- L/C * (np.exp(1j * phi))/(1+2j*L*((f-fr)/fr))))

def fit_resonance(f, power):

    # funzioni per il chi quadro
    def chi_square(A, B, C, D, QL, fr):
        return np.sum(((power - skewed_lorentzian(f, A, B, C, D, QL, fr))**2))

    def chi_square_2 (a, a0, a1, a2, a3, L, C, fr, phi, fmin) :
        return np.sum(((power - full_fit(f, a, a0, a1, a2, a3, L, C, fr, phi, fmin))**2))

    # inizializzo i parametri
    A0 = 1.0
    B0 = 1.0
    C0 = np.pi
    D0 = 1.0
    QL0 = 10.0
    fr0 = np.mean(f)

    # fit preliminare
    minuit = Minuit(chi_square, A=A0, B=B0, fr=fr0, C=C0, D=D0, QL=QL0)
    minuit.errordef = Minuit.LEAST_SQUARES
    minuit.migrad()
    minuit.hesse()

    # salvo i parametri
    params = minuit.values
    errors = minuit.errors
    print("Risultati del fit preliminare:")
    parametri = ["A", "B", "C", "D", "fr", "QL"]
    for parametro in parametri:
        print(f"{parametro} = {params[parametro]:.4f} ± {errors[parametro]:.4f}")
    
    plt.scatter(f, power, label='Dati', color='blue', s = 3)

    plt.plot(f, skewed_lorentzian(f, A = params["A"], B = params["B"], C = params["C"], D = params["D"], QL = params["QL"], fr = params["fr"]), color = "red")

    plt.xlabel('Frequenza (f)')
    plt.ylabel('|y|')
    plt.legend()
    plt.show()


    # eseguo il fit totale
    minuit2 = Minuit(chi_square_2, L= 820 ,a = 0.15- 0.075, a0 = 0, a1 = 10e-18 , a2 = 10e-27, a3 = 0.27,  fr=params["fr"], C=2000, phi=0.3, fmin=params["fr"])
    minuit2.fixed["fmin"] = True
    #minuit2.fixed["fr"] = True
    minuit2.limits["a3"] = (0, None)
    minuit2.limits["fr"] = (0,None)
    minuit2.limits["L"] = (0, None)
    minuit2.limits["C"] = (0, None)
    minuit2.limits["phi"] = (-np.pi, np.pi)
    #minuit2.fixed["L"] = True
    #minuit2.fixed["b"] = True
    #minuit2.fixed["a1"] = True
    minuit2.errordef = Minuit.LEAST_SQUARES
    minuit2.migrad()
    minuit2.hesse()

    # salvo i parametri
    params2 = minuit2.values 
    errors2 = minuit2.errors   

    return params2, errors2
    
    
    
    
    

# ---------------------------------------- FIT DEL CERCHIO -----------------------------------------

def calc_moments (x, y) :
    '''
    funzione per costruire la matrice dei momenti
    '''    
    x_sqrt = x * x
    y_sqrt = y * y
    z = x_sqrt + y_sqrt
    N_points = float(len(x))

    # somme
    x_sum = x.sum()
    y_sum = y.sum()
    z_sum = z.sum()
    xy_sum = (x*y).sum()
    xz_sum = (x*z).sum()
    yz_sum = (y*z).sum()

    # matrice
    M = np.array([ [(z*z).sum(), xz_sum, yz_sum, z_sum],  \
    [xz_sum, x_sqrt.sum(), xy_sum, x_sum], \
    [yz_sum, xy_sum, y_sqrt.sum(), y_sum], \
    [z_sum, x_sum, y_sum, N_points] ])

    return M


def calc_SVD(val,M):
    '''
    funzione che modifica la matrice M e calcola la decomposizione a valori singolari (SVD)
    '''            
    M[3][0] = M[3][0]+2*val
    M[0][3] = M[0][3]+2*val
    M[1][1] = M[1][1]-val
    M[2][2] = M[2][2]-val
    return np.linalg.svd(M)


def fit_circle (x, y) :
    '''
    funzione per trovare il centro e il raggio  del cerchio nel piano x-y
    '''
    M = calc_moments(x, y)
    
    # calcolo dei parametri a0, a1, a2 ,a3, a4
    a0 = ((M[2][0]*M[3][2]-M[2][2]*M[3][0])*M[1][1]-M[1][2]*M[2][0]*M[3][1]-M[1][0]*M[2][1]*M[3][2]+M[1][0]*M[2][2]*M[3][1]+M[1][2]*M[2][1]*M[3][0])*M[0][3]+(M[0][2]*M[2][3]*M[3][0]-M[0][2]*M[2][0]*M[3][3]+M[0][0]*M[2][2]*M[3][3]-M[0][0]*M[2][3]*M[3][2])*M[1][1]+(M[0][1]*M[1][3]*M[3][0]-M[0][1]*M[1][0]*M[3][3]-M[0][0]*M[1][3]*M[3][1])*M[2][2]+(-M[0][1]*M[1][2]*M[2][3]-M[0][2]*M[1][3]*M[2][1])*M[3][0]+((M[2][3]*M[3][1]-M[2][1]*M[3][3])*M[1][2]+M[2][1]*M[3][2]*M[1][3])*M[0][0]+(M[1][0]*M[2][3]*M[3][2]+M[2][0]*(M[1][2]*M[3][3]-M[1][3]*M[3][2]))*M[0][1]+((M[2][1]*M[3][3]-M[2][3]*M[3][1])*M[1][0]+M[1][3]*M[2][0]*M[3][1])*M[0][2]
    
    a1 = (((M[3][0]-2.*M[2][2])*M[1][1]-M[1][0]*M[3][1]+M[2][2]*M[3][0]+2.*M[1][2]*M[2][1]-M[2][0]*M[3][2])*M[0][3]+(2.*M[2][0]*M[3][2]-M[0][0]*M[3][3]-2.*M[2][2]*M[3][0]+2.*M[0][2]*M[2][3])*M[1][1]+(-M[0][0]*M[3][3]+2.*M[0][1]*M[1][3]+2.*M[1][0]*M[3][1])*M[2][2]+(-M[0][1]*M[1][3]+2.*M[1][2]*M[2][1]-M[0][2]*M[2][3])*M[3][0]+(M[1][3]*M[3][1]+M[2][3]*M[3][2])*M[0][0]+(M[1][0]*M[3][3]-2.*M[1][2]*M[2][3])*M[0][1]+(M[2][0]*M[3][3]-2.*M[1][3]*M[2][1])*M[0][2]-2.*M[1][2]*M[2][0]*M[3][1]-2.*M[1][0]*M[2][1]*M[3][2])
        
    a2 = ((2.*M[1][1]-M[3][0]+2.*M[2][2])*M[0][3]+(2.*M[3][0]-4.*M[2][2])*M[1][1]-2.*M[2][0]*M[3][2]+2.*M[2][2]*M[3][0]+M[0][0]*M[3][3]+4.*M[1][2]*M[2][1]-2.*M[0][1]*M[1][3]-2.*M[1][0]*M[3][1]-2.*M[0][2]*M[2][3])
        
    a3 = (-2.*M[3][0]+4.*M[1][1]+4.*M[2][2]-2.*M[0][3])
        
    a4 = -4.
    
    # polinomio di quarto grado f
    def func(x):
        return a0+a1*x+a2*x*x+a3*x*x*x+a4*x*x*x*x

    # derivata di f
    def d_func(x):
        return a1+2*a2*x+3*a3*x*x+4*a4*x*x*x

    # trovo una radice del polinomio
    x0 = spopt.fsolve(func, 0., fprime=d_func)
    
    # trovo le matrici della decomposizione e il vettore dei valori singolari s
    U,s,Vt = calc_SVD(x0[0],M)

    # estraggo il vettore associato al valore singolare minimo
    A_vec = Vt[np.argmin(s),:]

    # calcolo coordinate dle centro e raggio
    xc = -A_vec[1]/(2.*A_vec[0])
    yc = -A_vec[2]/(2.*A_vec[0])
    r = 1./(2.*np.absolute(A_vec[0]))*np.sqrt(A_vec[1]*A_vec[1]+A_vec[2]*A_vec[2]-4.*A_vec[0]*A_vec[3])
    return xc, yc, r
    



     
# -------------------------------- CALCOLO DI QL, Qc E Qi -----------------------------------

def calc_Q_factors (file_path) :
    '''
    funzione che calcola i fattori di qualità partendo da dati del tipo (frequenza, I, Q)
    '''
    # salvo i dati e calcolo il quadrato dell'ampiezza
    f, I, Q = read_data(file_path)
    power = I**2 + Q**2
    
    # faccio il fit della risonanza per ottenere QL
    params, errors = fit_resonance(f, power)
    QL = params[4]
    sigma_QL = errors[4]
    
    # faccio il fit del cerchio per ottenere Qc
    xc, yc, r = fit_circle(I, Q)
    Qc = (np.sqrt(xc**2 + yc**2) + r)/(2 * r) * QL
    
    # calcolo Qi
    Qi = 1/(1/QL - 1/Qc)

    return QL, Qi, Qc
