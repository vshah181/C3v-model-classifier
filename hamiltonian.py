import numpy as np

from kmesh import kfrac_to_kcart
from constants import (LATTICE_VECTORS, K00, KZ0, K01, K02, K03, KZ1, KZ2, KZ3, 
                    K0Z, KZZ, KX0, KYZ, KXZ, KY0, KY1, KY2, KY3)


def fixed_hamiltonian(kfrac, p):
    """
    Calculate the analytical hamiltonian over the mesh
    """
    kcart = kfrac_to_kcart(kfrac)

    k1 = np.dot(kcart, LATTICE_VECTORS[0])
    k2 = np.dot(kcart, LATTICE_VECTORS[1])
    k3 = -1.0 * (k1 + k2)
    kz_dot = np.dot(kcart, LATTICE_VECTORS[2])

    sk1 = np.sin(k1)
    sk2 = np.sin(k2)
    sk3 = np.sin(k3)
    skz = np.sin(kz_dot)

    ck1 = np.cos(k1)
    ck2 = np.cos(k2)
    ck3 = np.cos(k3)
    ckz = np.cos(kz_dot)

    c = p[0]
    mxy = p[1]
    mz = p[2]
    c_ = p[3]
    mxy_ = p[4]
    mz_ = p[5]
    # r = p[6]
    rxy = p[7]
    rz = p[8]
    # r_ = p[9]
    rxy_ = p[10]
    rz_ = p[11]
    q = p[12]
    q_ = p[13]
    h = p[14]
    h_ = p[15]
    a5 = p[16]
    a69 = p[17]
    a101 = p[18]
    a12 = p[19]
    a76 = p[20]
    a108 = p[21]
    a184 = p[22]
    a185 = p[23]
    a74 = p[24]
    a139 = p[25]

    disp_coeff = c + mxy * (3.0 - ck1 - ck2 - ck3)
    disp_coeff += mz * (1.0 + ckz)
    disp_part = disp_coeff * K00

    disp_coeff_ = c_ + mxy_ * (3.0 - ck1 - ck2 - ck3)
    disp_coeff_ += mz_ * (1.0 + ckz)
    disp_part_ = disp_coeff_ * KZ0

    spin_part_r =  rxy * (3.0 - ck1 - ck2 - ck3) + rz * (1.0 + ckz)
    spin_part_s =  sk1 * K01
    spin_part_s += sk2 * K02
    spin_part_s += sk3 * K03
    spin_part = spin_part_r * spin_part_s

    spin_part_r_ =  rxy_ * (3.0 - ck1 - ck2 - ck3) + rz_ * (1.0 + ckz)
    spin_part_s_ =  sk1 * KZ1
    spin_part_s_ += sk2 * KZ2
    spin_part_s_ += sk3 * KZ3
    spin_part_ = spin_part_r_ * spin_part_s_

    q_part = q *  skz * (ck1 - ck2) * K03
    q_part += q * skz * (ck2 - ck3) * K01
    q_part += q * skz * (ck3 - ck1) * K02

    q_part_ = q_  * skz * (ck1 - ck2) * KZ3
    q_part_ += q_ * skz * (ck2 - ck3) * KZ1
    q_part_ += q_ * skz * (ck3 - ck1) * KZ2

    h_coeff = sk1 * (1.0 - ck1) + sk2 * (1.0 - ck2)
    h_coeff += sk3 * (1.0 - ck3)
    h_part = h_coeff * h * K0Z
    
    h_part_ = h_coeff * h_ * KZZ

    first_a_coeff = a5 + a69 * (3.0 - ck1 - ck2 - ck3)
    first_a_coeff += a101 * (1 + ckz)
    a_part1 = first_a_coeff * KX0

    second_a_coeff = a12 + a76 * (3.0 - ck1 - ck2 - ck3)
    second_a_coeff += a108 * (1 + ckz)
    a_part2 = second_a_coeff * KYZ

    third_a_coeff = sk1 * (1.0 - ck1) + sk2 * (1.0 - ck2) + sk3 * (1.0 - ck3) 
    a_part3 = third_a_coeff * a184 * KXZ
    a_part3 += third_a_coeff * a185 * KY0

    fourth_a_coeff = a74 * (ck1 + ck1 - ck2 - ck3) * KY1
    fourth_a_coeff += a74 * (ck2 + ck2 - ck1 - ck3) * KY2
    fourth_a_coeff += a74 * (ck3 + ck3 - ck1 - ck2) * KY3
    fourth_a_coeff += a139 * (sk3 - sk2) * skz * KY1
    fourth_a_coeff += a139 * (sk1 - sk3) * skz * KY2
    fourth_a_coeff += a139 * (sk2 - sk1) * skz * KY3
    a_part4 = fourth_a_coeff

    hk = disp_part + disp_part_ + spin_part + spin_part_ + q_part + q_part_
    hk += h_part + h_part_ + a_part1 + a_part2 + a_part3 + a_part4

    return hk


def fixed_r_hamiltonian(kfrac):
    """
    Calculate the analytical hamiltonian over the mesh
    """
    kcart = kfrac_to_kcart(kfrac)

    k1 = np.dot(kcart, LATTICE_VECTORS[0])
    k2 = np.dot(kcart, LATTICE_VECTORS[1])
    k3 = -1.0 * (k1 + k2)

    sk1 = np.sin(k1)
    sk2 = np.sin(k2)
    sk3 = np.sin(k3)

    spin_part_s = sk1 * K01
    spin_part_s += sk2 * K02
    spin_part_s += sk3 * K03
    spin_part = spin_part_s

    hk = spin_part 
    return hk


def fixed_r__hamiltonian(kfrac):
    """
    Calculate the analytical hamiltonian over the mesh
    """
    kcart = kfrac_to_kcart(kfrac)

    k1 = np.dot(kcart, LATTICE_VECTORS[0])
    k2 = np.dot(kcart, LATTICE_VECTORS[1])
    k3 = -1.0 * (k1 + k2)

    sk1 = np.sin(k1)
    sk2 = np.sin(k2)
    sk3 = np.sin(k3)

    spin_part_s_ = sk1 * KZ1
    spin_part_s_ += sk2 * KZ2
    spin_part_s_ += sk3 * KZ3
    spin_part_ =  spin_part_s_

    hk =  spin_part_
    return hk


def hamiltonian(kfrac, r, r_, p):
    ham = fixed_hamiltonian(kfrac, p)
    ham += r * fixed_r_hamiltonian(kfrac)
    ham += r_ * fixed_r__hamiltonian(kfrac)
    return ham
