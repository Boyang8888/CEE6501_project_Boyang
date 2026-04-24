import numpy as np
from .geometry3d import transformation_matrix_12x12


def truss_element_kl_3d(E, A, L):
    """
    Return the 12x12 local stiffness matrix for a 3D truss element embedded
    in a 12-DOF frame-style vector:

    [ux, uy, uz, rx, ry, rz, ux, uy, uz, rx, ry, rz]

    Only local axial stiffness is active.
    """
    k_local = np.zeros((12, 12), dtype=float)

    a = E * A / L

    # local axial DOFs are ux_i and ux_j -> local indices 0 and 6
    axial_block = a * np.array(
        [
            [1.0, -1.0],
            [-1.0, 1.0],
        ],
        dtype=float,
    )

    k_local[np.ix_([0, 6], [0, 6])] = axial_block

    return k_local


def truss_element_Qf_3d():
    """
    First-version 3D truss element:
    no member loads, no temperature, no fabrication imperfection.
    """
    return np.zeros(12, dtype=float)


def truss_element_T_3d(R):
    return transformation_matrix_12x12(R)


def truss_axial_force_local(u_global, m_1based, T, E, A, L):
    """
    Recover local axial force N from local axial deformation.
    Tension positive.
    """
    idx = np.asarray(m_1based, dtype=int) - 1
    u_e_global = u_global[idx]
    u_e_local = T @ u_e_global

    du = u_e_local[6] - u_e_local[0]
    N = E * A / L * du
    return N

def truss_element_Qf_temp_3d(E, A, alpha, deltaT):
    """
    Uniform temperature change for truss element
    embedded in 3D frame DOFs.

    Return: 12x1 local fixed-end force vector
    DOF order:
    [uxi, uyi, uzi, rxi, ryi, rzi, uxj, uyj, uzj, rxj, ryj, rzj]
    """
    import numpy as np

    N0 = E * A * alpha * deltaT

    Qf = np.zeros(12, dtype=float)
    Qf[0] = -N0
    Qf[6] =  N0

    return Qf

def truss_element_Qf_fabrication_3d(E, A, L, deltaL):
    """
    Fabrication length error for truss element (embedded in 3D frame DOFs).
    deltaL < 0 means the member is shorter than intended.
    
    Return: 12x1 local fixed-end force vector (frame DOF order)
    DOF order: [uxi, uyi, uzi, rxi, ryi, rzi, uxj, uyj, uzj, rxj, ryj, rzj]
    """
    import numpy as np

    N0 = E * A * (deltaL / L)

    Qf = np.zeros(12, dtype=float)

    # axial force only (local x direction)
    Qf[0] = -N0   # uxi
    Qf[6] =  N0   # uxj

    return Qf