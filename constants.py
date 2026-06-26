import numpy as np
from user_options_local import LATTICE_VECTORS

# ============================================================
# CONSTANTS
# ============================================================

UNIT_CELL_VOLUME = np.dot(np.cross(LATTICE_VECTORS[0], LATTICE_VECTORS[1]), LATTICE_VECTORS[2])
PREF =  (np.pi + np.pi) / UNIT_CELL_VOLUME

RECIP_VECTORS = np.empty((3, 3), dtype=float)
RECIP_VECTORS[0] = PREF * np.cross(LATTICE_VECTORS[1], LATTICE_VECTORS[2])
RECIP_VECTORS[1] = PREF * np.cross(LATTICE_VECTORS[2], LATTICE_VECTORS[0])
RECIP_VECTORS[2] = PREF * np.cross(LATTICE_VECTORS[0], LATTICE_VECTORS[1])


SIGMA_0 = np.eye(2, dtype=complex)
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)

TAU_0 = np.eye(2, dtype=complex)
TAU_X = np.array([[0, 1], [1, 0]], dtype=complex)
TAU_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
TAU_Z = np.array([[1, 0], [0, -1]], dtype=complex)

SIGMA_1 = -0.5 * SIGMA_X - np.sqrt(0.75) * SIGMA_Y
SIGMA_2 = SIGMA_X
SIGMA_3 = -(SIGMA_1 + SIGMA_2)

# ------------------------------------------------------------
# Precompute all fixed Kronecker products once
# ------------------------------------------------------------

K00 = np.kron(TAU_0, SIGMA_0)
KZ0 = np.kron(TAU_Z, SIGMA_0)

K01 = np.kron(TAU_0, SIGMA_1)
K02 = np.kron(TAU_0, SIGMA_2)
K03 = np.kron(TAU_0, SIGMA_3)

KZ1 = np.kron(TAU_Z, SIGMA_1)
KZ2 = np.kron(TAU_Z, SIGMA_2)
KZ3 = np.kron(TAU_Z, SIGMA_3)

K0Z = np.kron(TAU_0, SIGMA_Z)
KZZ = np.kron(TAU_Z, SIGMA_Z)

KX0 = np.kron(TAU_X, SIGMA_0)
KYZ = np.kron(TAU_Y, SIGMA_Z)
KXZ = np.kron(TAU_X, SIGMA_Z)
KY0 = np.kron(TAU_Y, SIGMA_0)

KY1 = np.kron(TAU_Y, SIGMA_1)
KY2 = np.kron(TAU_Y, SIGMA_2)
KY3 = np.kron(TAU_Y, SIGMA_3)

def load_parameters(fname):
    return np.loadtxt(fname)
