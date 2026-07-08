import math
import numpy as np
from scipy.optimize import minimize

from user_options_local import TIGHT_TOLERANCE
from hamiltonian import hamiltonian


def gap_squared(kfrac, c_, a5, p):
    bands = np.linalg.eigvalsh(hamiltonian(kfrac, c_, a5, p))
    gap_squared = math.pow(bands[2] - bands[1], 2.0)
    return gap_squared


def find_nodes(kfrac, c_, a5, p):
    klim = [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]
    start = kfrac
    
    # Gradients around Weyl nodes can get messy, COBYQA maybe more robust?
    # But squaring the gap should address this (hopefully)
    result = minimize(fun=gap_squared, args=(c_, a5, p), x0=start, 
                      method="L-BFGS-B", bounds=klim)

    gap = np.sqrt(result.fun)
    found_node = (gap <= TIGHT_TOLERANCE)
    kpoint = result.x
    return found_node, gap, kpoint
