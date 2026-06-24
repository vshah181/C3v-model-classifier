import logging
import z2pack
from hamiltonian import hamiltonian
import warnings
import numpy as np


def needs_retry(warning_list):
    for w in warning_list:
        msg = str(w.message)
        if "Iterator stopped before the calculation could converge" in msg:
            return True
        elif "'min_neighbour_dist' reached: cannot add line at" in msg:
            return True
    return False


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


def get_z2_data(syst, surf):
    logging.getLogger("z2pack").setLevel(logging.CRITICAL)
    neighbour_dist = 1.0E-4
    n_lines = 12
    itr_num_start = 8
    itr_num_end = 27
    itr_num_step = 2

    for attempt in range(3):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = z2pack.surface.run(system=syst,
                                        min_neighbour_dist=neighbour_dist,
                                        surface=surf,
                                        num_lines=n_lines,
                                        iterator=range(itr_num_start,
                                                       itr_num_end,
                                                       itr_num_step))
            if not needs_retry(w):
                return result
            neighbour_dist *= 0.25
            n_lines += 6
            itr_num_end += 6

    """
    result = z2pack.surface.run(system=syst,
                                min_neighbour_dist=neighbour_dist,
                                surface=surf,
                                num_lines=n_lines,
                                pos_tol=pos_tlnc,
                                gap_tol=gap_tlnc,
                                move_tol=mov_tlnc)

    """
    return result


def get_z2_indices(hamiltonian_func, r, r_, parameters):
    z2pack_sys = initialise_z2pack_system(hamiltonian_func, r, r_, parameters)
    x0 = z2pack.invariant.z2(get_z2_data(z2pack_sys,
                                         lambda t1, t2: [0.0, t1/2, t2]))
    x1 = z2pack.invariant.z2(get_z2_data(z2pack_sys,
                                         lambda t1, t2: [0.5, t1/2, t2]))
    y1 = z2pack.invariant.z2(get_z2_data(z2pack_sys,
                                         lambda t1, t2: [t1/2, 0.5, t2]))
    z1 = z2pack.invariant.z2(get_z2_data(z2pack_sys,
                                         lambda t1, t2: [t1/2, t2, 0.5]))

    v0 = int((x0 + x1) % 2)
    v1 = int(x1)
    v2 = int(y1)
    v3 = int(z1)

    return [v0, v1, v2, v3]
