import numpy as np
from scipy.optimize import curve_fit

def linear(x, gradient, intercept):
    return ((gradient * x) + intercept)

def get_parameters():
    a5_topo = 5.779331821802257546e-02
    a12_topo = -2.518396617929931636e-01
    a5_triv = 1.913426825220057304e-01
    a12_triv = -2.008538110651625005e-02
    x = np.array([a5_topo, a5_triv])
    y = np.array([a12_topo, a12_triv])
    popt, pcov = curve_fit(linear, x, y)
    # print(f"gradient = {popt[0]:.8f}, intercept = {popt[1]:.8f}")
    return popt

