import numpy as np
import time

from kmesh import make_irreducible_klist
from hamiltonian import (fixed_hamiltonian, fixed_r_hamiltonian, 
                         fixed_r__hamiltonian, hamiltonian)
from search import find_nodes
from io_utils import write_nodes
from constants import load_parameters
from classification import get_weyl_chirality
from user_options import (NPAR_1, NPAR_2, PAR_1_MIN, PAR_1_MAX, PAR_2_MIN, 
                          PAR_2_MAX, LOOSE_TOLERANCE)


def main():
    start_time = time.time()
    parameters = load_parameters("fit_params.txt")

    kgrid_coarse = make_irreducible_klist()
    nkp = len(kgrid_coarse)

    fixed_hams = np.empty((nkp, 4, 4), dtype=complex)
    fixed_r_hams = np.empty((nkp, 4, 4), dtype=complex)
    fixed_r__hams = np.empty((nkp, 4, 4), dtype=complex)
    hams = np.empty((NPAR_1, NPAR_2, nkp, 4, 4), dtype=complex)

    for ik, kfrac in enumerate(kgrid_coarse):
        fixed_hams[ik] = fixed_hamiltonian(kfrac, parameters)
        fixed_r_hams[ik] = fixed_r_hamiltonian(kfrac)
        fixed_r__hams[ik] = fixed_r__hamiltonian(kfrac)
    
    
    r_vals = np.linspace(PAR_1_MIN, PAR_1_MAX, NPAR_1)
    r__vals = np.linspace(PAR_2_MIN, PAR_2_MAX, NPAR_2)
    hams = fixed_hams[None, None, :, :, :]\
         + r_vals[:, None, None, None, None] * fixed_r_hams[None, None, :, :, :]\
         + r__vals[None, :, None, None, None] * fixed_r__hams[None, None, :, :, :]

    eigenvals = np.linalg.eigvalsh(hams)
    differences = np.abs(eigenvals[..., 2] - eigenvals[..., 1])

    r_idx_all, r__idx_all, k_idx_all = np.where(differences < LOOSE_TOLERANCE)
    unique_candidates = {}
    print("Checking candidate points...")
    for i in range(len(r_idx_all)):
        r_idx = r_idx_all[i]
        r__idx = r__idx_all[i]
        r = r_vals[r_idx]
        r_ = r__vals[r__idx]

        k_idx = k_idx_all[i]
        k = kgrid_coarse[k_idx]

        node_found, gap, crossing_point = find_nodes(k, r, r_, parameters)
        if node_found:
            key = (r_idx, r__idx)
            cross_coord = crossing_point.copy()
            if key not in unique_candidates:
                unique_candidates[key] = {"gap": gap, "k": cross_coord, "chirality": 0}
            else:
                if gap < unique_candidates[key]["gap"]:
                    unique_candidates[key] = {"gap": gap, "k": cross_coord, "chirality": 0}

    """ 
    Now we have unique candidates all with a tiny gap.
    We must check the chirality. They are all 0 by default.
    """
    if unique_candidates:
        print("Found some points. Checking their chiralities...")
        for (r_idx, r__idx), info in unique_candidates.items():
            r = r_vals[r_idx]
            r_ = r__vals[r__idx]
            gap = info["gap"]
            info["chirality"] = get_weyl_chirality(hamiltonian, r, r_,
                                                   parameters, info["k"])

    if unique_candidates:
        write_nodes(unique_candidates, r_vals, r__vals)
    else:
        print("Couldn't find any Weyl nodes")

    print(f"took {(time.time() - start_time):.1f} seconds.")

if __name__ == "__main__":
    main()
