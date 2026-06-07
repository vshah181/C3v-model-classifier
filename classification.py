import logging
import z2pack


def make_z2pack_hamiltonian(hamiltonian_func, r, r_, p):
    def hk(kfrac):
        return hamiltonian_func(kfrac, r, r_, p)
    return hk


def initialise_z2pack_system(hamiltonian_func, r, r_, p, num_occ=2):
    hk = make_z2pack_hamiltonian(hamiltonian_func, r, r_, p)
    system = z2pack.hm.System(hk, bands=num_occ)
    return system


def get_weyl_chirality(hamiltonian_func, r, r_, parameters, kpoint, radius=0.005):
    logging.getLogger("z2pack").setLevel(logging.WARNING) # Very annoying otherwise

    z2pack_sys = initialise_z2pack_system(hamiltonian_func, r, r_, parameters)
    result = z2pack.surface.run(system=z2pack_sys, 
                                surface=z2pack.shape.Sphere(kpoint, radius),
                                save_file=None, load=None,
                                min_neighbour_dist=5.0E-4)
    return z2pack.invariant.chern(result)

