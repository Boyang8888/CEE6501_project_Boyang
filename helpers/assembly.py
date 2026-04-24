import numpy as np
from .frame3d import frame3d_element_equivalent_nodal_load_global

def element_dof_map_3d_1based(node_i, node_j):
    """
    1-based global DOF numbering for 3D frame/truss:
    node n -> [6n-5, 6n-4, 6n-3, 6n-2, 6n-1, 6n]
    """
    ni = int(node_i)
    nj = int(node_j)

    m_i = [6 * ni - 5, 6 * ni - 4, 6 * ni - 3, 6 * ni - 2, 6 * ni - 1, 6 * ni]
    m_j = [6 * nj - 5, 6 * nj - 4, 6 * nj - 3, 6 * nj - 2, 6 * nj - 1, 6 * nj]

    return np.array(m_i + m_j, dtype=int)


def assemble_global_stiffness_and_fef(
    ndof,
    k_list,
    T_list,
    Qf_list,
    map_list,
):
    """
    Assemble the global stiffness matrix and the global fixed-end force vector.
    Keeps the same naming style as the original 2D code.
    """
    K_global = np.zeros((ndof, ndof), dtype=float)
    F_fef_global = np.zeros(ndof, dtype=float)

    nelem = len(k_list)

    for i in range(nelem):
        k_local = k_list[i]
        T = T_list[i]
        Qf_local = Qf_list[i]
        dof_map = map_list[i]  # 1-based indexing

        edof = k_local.shape[0]

        K = T.T @ k_local @ T
        F_fef = T.T @ Qf_local

        for a in range(edof):
            A = dof_map[a] - 1
            F_fef_global[A] += F_fef[a]

            for b in range(edof):
                B = dof_map[b] - 1
                K_global[A, B] += K[a, b]

    return K_global, F_fef_global

def assemble_global_member_load_vector(
    elements,
    member_loads,
    element_length_map,
    element_T_map,
    element_dof_map,
    total_dof
):
    """
    Assemble equivalent nodal loads from frame member loads into a global vector.

    Parameters
    ----------
    elements : dict
        Element definitions.
    member_loads : dict[int, list[dict]]
        Element member loads.
    element_length_map : dict[int, float]
        Element length by element id.
    element_T_map : dict[int, ndarray]
        Element transformation matrix by element id.
    element_dof_map : dict[int, list[int]]
        Global DOF map for each element (0-based indices preferred).
    total_dof : int
        Total number of DOFs in the structure.

    Returns
    -------
    F_fef_global : (total_dof,) ndarray
        Global equivalent nodal load vector from member loads.
    """
    F_fef_global = np.zeros(total_dof, dtype=float)

    for e_id in elements:
        load_list = member_loads.get(e_id, [])
        if not load_list:
            continue

        L = element_length_map[e_id]
        T = element_T_map[e_id]
        dofs = element_dof_map[e_id]

        p_global_e = frame3d_element_equivalent_nodal_load_global(load_list, L, T)

        for a in range(len(dofs)):
            A = dofs[a]
            F_fef_global[A] += p_global_e[a]

    return F_fef_global