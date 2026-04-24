import numpy as np


def partition_system(K, f, u, f_fef, dof_restrained_1based):
    """
    Same structure and naming style as the 2D code.
    """
    ndof = K.shape[0]

    restrained_dofs = sorted(int(d) - 1 for d in dof_restrained_1based)
    free_dofs = [i for i in range(ndof) if i not in restrained_dofs]

    K_ff = K[np.ix_(free_dofs, free_dofs)]
    K_fr = K[np.ix_(free_dofs, restrained_dofs)]
    K_rf = K[np.ix_(restrained_dofs, free_dofs)]
    K_rr = K[np.ix_(restrained_dofs, restrained_dofs)]

    f_f = f[free_dofs]
    f_r = f[restrained_dofs]

    u_r = u[restrained_dofs]

    f_fef_f = f_fef[free_dofs]
    f_fef_r = f_fef[restrained_dofs]

    return (
        K_ff,
        K_fr,
        K_rf,
        K_rr,
        f_f,
        f_r,
        u_r,
        f_fef_f,
        f_fef_r,
        free_dofs,
        restrained_dofs,
    )


def assemble_global_displacements(u_f, u_r, free_dofs, restrained_dofs):
    ndof_total = len(free_dofs) + len(restrained_dofs)
    u_global = np.zeros(ndof_total, dtype=float)

    if u_r is None:
        u_r = np.zeros(len(restrained_dofs), dtype=float)

    u_global[free_dofs] = u_f
    u_global[restrained_dofs] = u_r

    return u_global


def assemble_global_forces(f_f, F_r, free_dofs, restrained_dofs):
    ndof_total = len(free_dofs) + len(restrained_dofs)
    f_global_complete = np.zeros(ndof_total, dtype=float)

    f_global_complete[free_dofs] = f_f
    f_global_complete[restrained_dofs] = F_r

    return f_global_complete


def solve_linear_static(K_global, F_global, u_global, F_fef_global, dof_restrained_1based):
    """
    Linear static solver using equivalent nodal loads from member loads.

    Governing system:
        K_global u_global = F_global + F_fef_global
    """
    (
        K_ff,
        K_fr,
        K_rf,
        K_rr,
        f_f,
        f_r,
        u_r,
        f_fef_f,
        f_fef_r,
        free_dofs,
        restrained_dofs,
    ) = partition_system(
        K_global,
        F_global,
        u_global,
        F_fef_global,
        dof_restrained_1based,
    )

    rhs = f_f + f_fef_f - K_fr @ u_r
    u_f = np.linalg.solve(K_ff, rhs)

    u_global = assemble_global_displacements(u_f, u_r, free_dofs, restrained_dofs)

    F_r = K_rf @ u_f + K_rr @ u_r - f_fef_r
    f_global_complete = assemble_global_forces(f_f + f_fef_f, F_r, free_dofs, restrained_dofs)

    return {
        "u_global": u_global,
        "f_global_complete": f_global_complete,
        "F_r": F_r,
        "K_ff": K_ff,
        "free_dofs": free_dofs,
        "restrained_dofs": restrained_dofs,
    }