import numpy as np
import kmesh
from hamiltonian import (fixed_hamiltonian, fixed_c__hamiltonian, 
                         fixed_a5_hamiltonian, fixed_a12_hamiltonian, hamiltonian)
from user_options_local import (NPAR_1, NPAR_2, PAR_1_MIN, PAR_2_MIN,
                                PAR_1_MAX, PAR_2_MAX, INTERCEPT, GRADIENT)
from constants import load_parameters
from search import find_nodes
from classification import get_weyl_chirality
import io_utils


def read_error(fname="phase.err"):
    p1_list = []
    p2_list = []
    status_list = []
    with open(fname, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "At point" in line:
                split_line = line.split()
                p1 = float(split_line[-5])
                p2 = float(split_line[-4])
                status = float(split_line[-1])
                p1_list.append(p1)
                p2_list.append(p2)
                status_list.append(status)
    return np.transpose(np.array([p1_list, p2_list, status_list]))


def read_heatmap(fname="phase_diagram.csv"):
    return np.loadtxt(fname, delimiter=",")


def coarse_search(param_pair, parameters):
    c_, a5 = param_pair
    a12 = (GRADIENT * a5) + INTERCEPT

    kgrid_coarse = kmesh.make_irreducible_klist()
    nkp = len(kgrid_coarse)

    hamiltonians = np.empty((nkp, 4, 4), dtype=complex)

    for ik, kfrac in enumerate(kgrid_coarse):
        hamiltonians[ik] = fixed_hamiltonian(kfrac, parameters)
        hamiltonians[ik] += c_ * fixed_c__hamiltonian(kfrac)
        hamiltonians[ik] += a5 * fixed_a5_hamiltonian(kfrac)
        hamiltonians[ik] += a12 * fixed_a12_hamiltonian(kfrac)

    eigenvalues = np.linalg.eigvalsh(hamiltonians)
    differences = np.abs(eigenvalues[..., 2] - eigenvalues[..., 1])
    k_idx = np.argmin(differences)
    return kgrid_coarse[k_idx]


def bust_the_sus():
    sus_points = read_error("error.txt")
    all_points = read_heatmap("phase_diagram.csv")
    all_pairs = all_points[:, 0:2]

    parameters = load_parameters("fit_params.txt")
    for sus_index in range(len(sus_points)):
        target_pair = sus_points[sus_index, 0:2]
        status = sus_points[sus_index, -1]

        heatmap_index = np.argmin(np.sum((all_pairs - target_pair) ** 2, axis=1))

        if (status == 0):
            kpoint_coarse = coarse_search(target_pair, parameters)
            c_, a5 = target_pair
            node_found, gap, kpoint = find_nodes(kpoint_coarse, c_, a5, parameters)
            if node_found:
                chirlatiy = get_weyl_chirality(hamiltonian, c_, a5, parameters, kpoint)
                if 0.9 < abs(chirlatiy) < 1.1:
                    all_points[heatmap_index, -1] = 4
                else:
                    all_points[heatmap_index, -1] = 0
            else:
                all_points[heatmap_index, -1] = 0
    return all_points


if __name__ == "__main__":
    updated_heatmap = bust_the_sus()
    c__vals = np.linspace(PAR_1_MIN, PAR_1_MAX, NPAR_1)
    a5_vals = np.linspace(PAR_2_MIN, PAR_2_MAX, NPAR_2)
    reshaped_heatmap = np.empty((NPAR_1, NPAR_2))
    ip = 0
    for ic in range(len(c__vals)):
        for ia5 in range(len(a5_vals)):
            reshaped_heatmap[ic, ia5] = updated_heatmap[ip, -1]
            ip += 1
    io_utils.write_heatmap(c__vals, a5_vals, reshaped_heatmap, "sus_busted.csv")
