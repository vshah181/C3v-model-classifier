import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import os

from kmesh import make_irreducible_klist
from hamiltonian import (fixed_hamiltonian, fixed_c__hamiltonian, 
                         fixed_a12_hamiltonian, hamiltonian)
from search import find_nodes
from io_utils import write_nodes, write_heatmap
from constants import load_parameters
from classification import get_weyl_chirality, classify_pair
from plot_phase import plot_phase_diagram
from user_options_local import (NPAR_1, NPAR_2, PAR_1_MIN, PAR_1_MAX,
                          PAR_2_MIN, PAR_2_MAX, LOOSE_TOLERANCE)


def main():
    start_time = time.time()
    parameters = load_parameters("fit_params.txt")

    kgrid_coarse = make_irreducible_klist()
    nkp = len(kgrid_coarse)

    fixed_hams = np.empty((nkp, 4, 4), dtype=complex)
    fixed_c__hams = np.empty((nkp, 4, 4), dtype=complex)
    fixed_a12_hams = np.empty((nkp, 4, 4), dtype=complex)
    hams = np.empty((NPAR_1, NPAR_2, nkp, 4, 4), dtype=complex)

    for ik, kfrac in enumerate(kgrid_coarse):
        fixed_hams[ik] = fixed_hamiltonian(kfrac, parameters)
        fixed_c__hams[ik] = fixed_c__hamiltonian(kfrac)
        fixed_a12_hams[ik] = fixed_a12_hamiltonian(kfrac)
    
    
    c__vals = np.linspace(PAR_1_MIN, PAR_1_MAX, NPAR_1)
    a12_vals = np.linspace(PAR_2_MIN, PAR_2_MAX, NPAR_2)
    heatmap = np.zeros((NPAR_1, NPAR_2))
    hams = fixed_hams[None, None, :, :, :]\
         + c__vals[:, None, None, None, None] * fixed_c__hams[None, None, :, :, :]\
         + a12_vals[None, :, None, None, None] * fixed_a12_hams[None, None, :, :, :]

    eigenvals = np.linalg.eigvalsh(hams)
    differences = np.abs(eigenvals[..., 2] - eigenvals[..., 1])

    c__idx_all, a12_idx_all, k_idx_all = np.where(differences < LOOSE_TOLERANCE)
    unique_candidates = {}
    print("Checking candidate points...")
    for i in range(len(c__idx_all)):
        c__idx = c__idx_all[i]
        a12_idx = a12_idx_all[i]
        c_ = c__vals[c__idx]
        a12 = a12_vals[a12_idx]

        k_idx = k_idx_all[i]
        k = kgrid_coarse[k_idx]

        node_found, gap, crossing_point = find_nodes(k, c_, a12, parameters)
        if node_found:
            key = (c__idx, a12_idx)
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
        for (c__idx, a12_idx), info in unique_candidates.items():
            c_ = c__vals[c__idx]
            a12 = a12_vals[a12_idx]
            gap = info["gap"]
            info["chirality"] = get_weyl_chirality(hamiltonian, c_, a12,
                                                   parameters, info["k"])

    if unique_candidates:
        write_nodes(unique_candidates, c__vals, a12_vals)
    else:
        print("Couldn't find any Weyl nodes")

    # Now we need to look at the rest of the (r, r') pairs
    # loop through and ditch the ones with Weyl?

    print("Now checking Z2 indices")
    c__a12_index_pairs = np.empty((NPAR_1 * NPAR_2, 2), dtype=int)
    ikey = 0
    for ic_ in range(len(c__vals)):
        for ia12 in range(len(a12_vals)):
            c__a12_index_pairs[ikey, 0] = ic_
            c__a12_index_pairs[ikey, 1] = ia12
            ikey += 1

    n_workers = int(os.environ.get("MY_NUM_WORKERS", 6))
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # submit all jobs in parallel
        results = executor.map(classify_pair, 
                               map(tuple, c__a12_index_pairs),
                               repeat(c__vals),
                               repeat(a12_vals),
                               repeat(unique_candidates),
                               repeat(parameters))
        for ic_, ia12, value in results:
            heatmap[ic_, ia12] = value

    write_heatmap(c__vals, a12_vals, heatmap, fname="phase_diagram.csv")
    plot_phase_diagram(fname="phase_diagram.csv", delim=",")

    print(f"took {(time.time() - start_time):.1f} seconds.")

if __name__ == "__main__":
    main()
