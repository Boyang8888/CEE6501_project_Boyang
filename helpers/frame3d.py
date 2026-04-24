import numpy as np
from .geometry3d import transformation_matrix_12x12


def frame_element_kl_3d(E, G, A, Iy, Iz, J, L):
    """
    Return the 12x12 local stiffness matrix for a 3D frame element.

    Local DOF ordering
    ------------------
    [ux, uy, uz, rx, ry, rz, ux, uy, uz, rx, ry, rz]
    """
    k = np.zeros((12, 12), dtype=float)

    # ------------------------------------------------------------
    # 1) Axial
    # ------------------------------------------------------------
    EA_L = E * A / L
    k[np.ix_([0, 6], [0, 6])] += EA_L * np.array(
        [
            [1.0, -1.0],
            [-1.0, 1.0],
        ],
        dtype=float,
    )

    # ------------------------------------------------------------
    # 2) Torsion
    # ------------------------------------------------------------
    GJ_L = G * J / L
    k[np.ix_([3, 9], [3, 9])] += GJ_L * np.array(
        [
            [1.0, -1.0],
            [-1.0, 1.0],
        ],
        dtype=float,
    )

    # ------------------------------------------------------------
    # 3) Bending about local z-axis
    #    couples: local y translation + local rz rotation
    #    DOFs: [uy_i, rz_i, uy_j, rz_j] = [1, 5, 7, 11]
    # ------------------------------------------------------------
    EIz = E * Iz
    L2 = L * L
    L3 = L2 * L

    kz = np.array(
        [
            [12.0 * EIz / L3,   6.0 * EIz / L2,  -12.0 * EIz / L3,   6.0 * EIz / L2],
            [ 6.0 * EIz / L2,   4.0 * EIz / L,   -6.0 * EIz / L2,   2.0 * EIz / L ],
            [-12.0 * EIz / L3, -6.0 * EIz / L2,   12.0 * EIz / L3, -6.0 * EIz / L2],
            [ 6.0 * EIz / L2,   2.0 * EIz / L,   -6.0 * EIz / L2,   4.0 * EIz / L ],
        ],
        dtype=float,
    )

    dofs_z = [1, 5, 7, 11]
    k[np.ix_(dofs_z, dofs_z)] += kz

    # ------------------------------------------------------------
    # 4) Bending about local y-axis
    #    couples: local z translation + local ry rotation
    #    DOFs: [uz_i, ry_i, uz_j, ry_j] = [2, 4, 8, 10]
    # ------------------------------------------------------------
    EIy = E * Iy

    ky = np.array(
        [
            [12.0 * EIy / L3,  -6.0 * EIy / L2,  -12.0 * EIy / L3,  -6.0 * EIy / L2],
            [-6.0 * EIy / L2,   4.0 * EIy / L,    6.0 * EIy / L2,   2.0 * EIy / L ],
            [-12.0 * EIy / L3,  6.0 * EIy / L2,   12.0 * EIy / L3,   6.0 * EIy / L2],
            [-6.0 * EIy / L2,   2.0 * EIy / L,    6.0 * EIy / L2,   4.0 * EIy / L ],
        ],
        dtype=float,
    )

    dofs_y = [2, 4, 8, 10]
    k[np.ix_(dofs_y, dofs_y)] += ky

    return k


def frame_element_T_3d(R):
    return transformation_matrix_12x12(R)


def frame_element_Qf_3d(L, e_loads):
    """
    3D frame element fixed-end force vector in local coordinates.

    Local DOF order:
    [ux_i, uy_i, uz_i, rx_i, ry_i, rz_i, ux_j, uy_j, uz_j, rx_j, ry_j, rz_j]
    """
    Qf = np.zeros(12, dtype=float)

    for load in e_loads:
        load_type = load["type"]

        # --------------------------------------------------
        # 1) Uniform distributed load
        # --------------------------------------------------
        if load_type == "udl_local":
            qx = float(load.get("qx", 0.0))
            qy = float(load.get("qy", 0.0))
            qz = float(load.get("qz", 0.0))

            # axial
            if qx != 0.0:
                Fx = qx * L / 2.0
                Qf[0] += Fx
                Qf[6] += Fx

            # transverse in local y -> shear Fy and moment Mz
            if qy != 0.0:
                Fy = qy * L / 2.0
                Mz = qy * L**2 / 12.0

                Qf[1] += Fy
                Qf[7] += Fy
                Qf[5] += Mz
                Qf[11] -= Mz

            # transverse in local z -> shear Fz and moment My
            if qz != 0.0:
                Fz = qz * L / 2.0
                My = qz * L**2 / 12.0

                Qf[2] += Fz
                Qf[8] += Fz
                Qf[4] -= My
                Qf[10] += My

        # --------------------------------------------------
        # 2) Concentrated point load
        # --------------------------------------------------
        elif load_type == "point_local":
            Px = float(load.get("Px", 0.0))
            Py = float(load.get("Py", 0.0))
            Pz = float(load.get("Pz", 0.0))
            a = float(load["a"])
            b = L - a

            # axial point load
            if Px != 0.0:
                Fx_i = Px * b / L
                Fx_j = Px * a / L
                Qf[0] += Fx_i
                Qf[6] += Fx_j

            # point load in local y -> Fy, Mz
            if Py != 0.0:
                Fy_i = Py * b**2 * (3*a + b) / L**3
                Fy_j = Py * a**2 * (a + 3*b) / L**3
                Mz_i = Py * a * b**2 / L**2
                Mz_j = - Py * a**2 * b / L**2

                Qf[1] += Fy_i
                Qf[7] += Fy_j
                Qf[5] += Mz_i
                Qf[11] += Mz_j

            # point load in local z -> Fz, My
            if Pz != 0.0:
                Fz_i = Pz * b**2 * (3*a + b) / L**3
                Fz_j = Pz * a**2 * (a + 3*b) / L**3
                My_i = Pz * a * b**2 / L**2
                My_j = -Pz * a**2 * b / L**2

                Qf[2] += Fz_i
                Qf[8] += Fz_j
                Qf[4] += My_i
                Qf[10] += My_j

        else:
            raise ValueError(f"Unsupported member load type: {load_type}")

    return Qf


def frame_element_end_forces_local(u_global, m_1based, T, k_local, Qf_local):
    """
    Recover local element end forces:
        q_local = k_local @ v_local - Qf_local
    """
    idx = np.asarray(m_1based, dtype=int) - 1
    u_e_global = u_global[idx]
    v_local = T @ u_e_global
    q_local = k_local @ v_local - Qf_local
    return v_local, q_local



def frame3d_udl_equivalent_nodal_load_local(qx, qy, qz, L):
    """
    Equivalent nodal load vector in LOCAL coordinates for a 3D frame element
    under uniformly distributed loads qx, qy, qz along the local axes.

    Local DOF order assumed:
    [u1, v1, w1, rx1, ry1, rz1, u2, v2, w2, rx2, ry2, rz2]

    Parameters
    ----------
    qx, qy, qz : float
        Uniformly distributed loads in local x, y, z directions
        (force / length).
    L : float
        Element length.

    Returns
    -------
    p_local : (12,) ndarray
        Equivalent nodal load vector in local coordinates.
        This is the load vector to ADD to the global external load vector
        after transformation to global coordinates.
    """
    p = np.zeros(12, dtype=float)

    # Axial distributed load along local x
    if qx != 0.0:
        p += np.array([
            qx * L / 2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            qx * L / 2.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=float)

    # Transverse distributed load along local y
    # Produces bending about local z
    if qy != 0.0:
        p += np.array([
            0.0, qy * L / 2.0, 0.0, 0.0, 0.0, qy * L**2 / 12.0,
            0.0, qy * L / 2.0, 0.0, 0.0, 0.0, -qy * L**2 / 12.0
        ], dtype=float)

    # Transverse distributed load along local z
    # Produces bending about local y
    if qz != 0.0:
        p += np.array([
            0.0, 0.0, qz * L / 2.0, 0.0, -qz * L**2 / 12.0, 0.0,
            0.0, 0.0, qz * L / 2.0, 0.0,  qz * L**2 / 12.0, 0.0
        ], dtype=float)

    return p

def frame3d_element_equivalent_nodal_load_local(load_list, L):
    """
    Sum all supported member loads on one element into one local equivalent
    nodal load vector.

    Parameters
    ----------
    load_list : list[dict]
        List of load definitions for one element.
    L : float
        Element length.

    Returns
    -------
    p_local : (12,) ndarray
    """
    p_local = np.zeros(12, dtype=float)

    if not load_list:
        return p_local

    for load in load_list:
        load_type = load["type"]

        if load_type == "udl_local":
            qx = float(load.get("qx", 0.0))
            qy = float(load.get("qy", 0.0))
            qz = float(load.get("qz", 0.0))
            p_local += frame3d_udl_equivalent_nodal_load_local(qx, qy, qz, L)
        else:
            raise ValueError(f"Unsupported frame member load type: {load_type}")

    return p_local

def frame3d_element_equivalent_nodal_load_global(load_list, L, T):
    """
    Transform element equivalent nodal loads from local to global coordinates.

    Parameters
    ----------
    load_list : list[dict]
        Member loads on this element.
    L : float
        Element length.
    T : (12,12) ndarray
        Element transformation matrix such that:
            d_local = T @ d_global

    Returns
    -------
    p_global : (12,) ndarray
        Equivalent nodal load vector in global coordinates.
    """
    p_local = frame3d_element_equivalent_nodal_load_local(load_list, L)
    p_global = T.T @ p_local
    return p_global


def frame_element_Qf_temp_3d(E, A, alpha, deltaT):
    Nth = E * A * alpha * deltaT
    return np.array(
        [-Nth, 0.0, 0.0, 0.0, 0.0, 0.0,
          Nth, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=float
    )


def frame_element_Qf_fabrication_3d(E, A, L, deltaL):
    """
    3D frame element local fixed-end force vector due to fabrication length error.
    Axial effect only.

    Local DOF order:
    [u1, v1, w1, rx1, ry1, rz1, u2, v2, w2, rx2, ry2, rz2]
    """
    N = E * A * deltaL / L

    Qf = np.zeros(12, dtype=float)
    Qf[0] = -N
    Qf[6] =  N
    return Qf