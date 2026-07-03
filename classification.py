import logging
import sys
import z2pack
from hamiltonian import hamiltonian
import numpy as np


def needs_retry(convergence_report):

    def check_block(block):
        return all(len(v.get("FAILED", [])) == 0 for v in block.values())

    return not(
        check_block(convergence_report.get("line", {})) and
        check_block(convergence_report.get("surface", {}))
    )

def classify_pair(idx_pair, r_vals, r__vals, unique_candidates, parameters):
    ir = idx_pair[0]
    ir_ = idx_pair[1]

    r = r_vals[ir]
    r_ = r__vals[ir_]
    key = (ir, ir_)
    if key not in unique_candidates:
        z2_indices = get_z2_indices(hamiltonian, r, r_, parameters)
        strong_index = z2_indices[0]
        weak_indices = np.array(z2_indices[1:])
        if strong_index != 0:
            result = 3
        elif any(weak_indices != 0):
            result = 2
        else:
            result = 1
    elif 0.9 < abs(unique_candidates[key]["chirality"]) < 1.1:
        result = 4
    else:
        result = 0
    return ir, ir_, result


def make_z2pack_hamiltonian(hamiltonian_func, r, r_, p):
    def hk(kfrac):
        return hamiltonian_func(kfrac, r, r_, p)
    return hk


def initialise_z2pack_system(hamiltonian_func, r, r_, p, num_occ=2):
    hk = make_z2pack_hamiltonian(hamiltonian_func, r, r_, p)
    system = z2pack.hm.System(hk, bands=num_occ)
    return system


def get_weyl_chirality(hamiltonian_func, r, r_, parameters, kpoint, radius=0.005):
    logging.getLogger("z2pack").setLevel(logging.CRITICAL) # Very annoying otherwise

    z2pack_sys = initialise_z2pack_system(hamiltonian_func, r, r_, parameters)
    result = z2pack.surface.run(system=z2pack_sys, 
                                surface=z2pack.shape.Sphere(kpoint, radius),
                                save_file=None, load=None,
                                min_neighbour_dist=1.0E-4)
    return z2pack.invariant.chern(result)


def get_z2_data(syst, surf, r, r_):
    logging.getLogger("z2pack").setLevel(logging.CRITICAL)
    neighbour_dist = 1.0E-4
    n_lines = 12
    itr_num_start = 8
    itr_num_end = 27
    itr_num_step = 2

    for attempt in range(5):
        result = z2pack.surface.run(system=syst,
                                    min_neighbour_dist=neighbour_dist,
                                    surface=surf,
                                    num_lines=n_lines,
                                    iterator=range(itr_num_start,
                                                   itr_num_end,
                                                   itr_num_step))
        if not needs_retry(result.convergence_report):
            return result
        
        remark_message = "REMARK: could not converge on the first try. "
        remark_message += "Trying again with different parameters..."
        print(remark_message)
        neighbour_dist *= 0.25
        n_lines += 6
        itr_num_end += 6

    warning_message = "Warning, there may have been a Z2Pack problem! "
    warning_message += f"At point {r:.6f} {r_:.6f}.\n"
    sys.stderr.write(warning_message)
    return result


def get_z2_indices(hamiltonian_func, r, r_, parameters):
    z2pack_sys = initialise_z2pack_system(hamiltonian_func, r, r_, parameters)
    x0 = z2pack.invariant.z2(get_z2_data(z2pack_sys,
                                         lambda t1, t2: [0.0, t1/2, t2],
                                         r, r_))
    x1 = z2pack.invariant.z2(get_z2_data(z2pack_sys,
                                         lambda t1, t2: [0.5, t1/2, t2],
                                         r, r_))
    y1 = z2pack.invariant.z2(get_z2_data(z2pack_sys,
                                         lambda t1, t2: [t1/2, 0.5, t2],
                                         r, r_))
    z1 = z2pack.invariant.z2(get_z2_data(z2pack_sys,
                                         lambda t1, t2: [t1/2, t2, 0.5],
                                         r, r_))

    v0 = int((x0 + x1) % 2)
    v1 = int(x1)
    v2 = int(y1)
    v3 = int(z1)

    return [v0, v1, v2, v3]
