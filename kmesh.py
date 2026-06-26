import numpy as np
import spglib
from constants import RECIP_VECTORS
from user_options_local import LATTICE_VECTORS, NKX, NKY, NKZ, POSITIONS

def make_irreducible_klist():
    nk1 = NKX
    nk2 = NKY
    nk3 = NKZ

    lattice_vectors = LATTICE_VECTORS.copy()
    positions = POSITIONS.copy()

    numbers = [1, 2, 3]
    cell = (lattice_vectors, positions, numbers)
    mesh = np.array([nk1, nk2, nk3])
    is_shift = np.zeros(3)
    mapping_table, grid_address = spglib.get_ir_reciprocal_mesh(mesh, cell,
                                                                is_shift=is_shift,
                                                                is_time_reversal=True, 
                                                                symprec=1e-5)
    ir_kpoints_frac = []
    for i in range(len(grid_address)):
        if mapping_table[i] == i:
            kfrac = (grid_address[i] + is_shift) / mesh
            ir_kpoints_frac.append(kfrac)
    return np.array(ir_kpoints_frac)


def kfrac_to_kcart(kfrac):
    kcart = np.zeros((3))
    for i in range(3):
        kcart += kfrac[i] * RECIP_VECTORS[i]
    return kcart
