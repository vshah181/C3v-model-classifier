# ------------------------------------------------------------
# User parameters
# ------------------------------------------------------------
LOOSE_TOLERANCE = 0.10000
TIGHT_TOLERANCE = 0.0001

NKX = 41
NKY = 41
NKZ = 23

NPAR_1 = 30
NPAR_2 = 30

PAR_1_MIN = -0.5
PAR_1_MAX =  0.5
PAR_2_MIN = -0.5
PAR_2_MAX =  0.5

LATTICE_VECTORS = np.array([[3.5588157, -2.0546835, 0.0000000],
                            [0.0000000,  4.1093664, 0.0000000],
                            [0.0000000,  0.0000000, 6.4193768]], dtype=float)

# Completely arbitrary. Just want C3v point symmetry.
POSITIONS = np.array([[0.0000000000, 0.0000000000, 0.46322936],
                      [0.3333333333, 0.6666666667, 0.21480074],
                      [0.6666666667, 0.3333333333, 0.76576990]], dtype=float)
