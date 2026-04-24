import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os



def plot_truss_3d_plotly(
    nodes,
    elements,
    nodes_restrained=None,
    nodes_loaded=None,
    show_node_ids=True,
    show_member_ids=False,
    load_scale=0.02,
):
    """
    nodes: {nid: (x,y,z)}
    elements: {eid: (i,j,E,A)}
    nodes_restrained: {nid: ["ux","uy","uz"], ...}
    nodes_loaded: {nid: (Fx,Fy,Fz)}
    """

    # ---- normalize nodes to 3D ----
    nodes3 = {}
    for nid, xyz in nodes.items():
        if len(xyz) == 2:
            x, y = xyz
            z = 0.0
        else:
            x, y, z = xyz
        nodes3[nid] = (float(x), float(y), float(z))

    fig = go.Figure()

    # =========================================================
    # Members (lines)
    # =========================================================
    xs, ys, zs = [], [], []
    for eid, (i, j, E, A) in elements.items():
        xi, yi, zi = nodes3[i]
        xj, yj, zj = nodes3[j]
        xs += [xi, xj, None]
        ys += [yi, yj, None]
        zs += [zi, zj, None]

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(width=3),
            name="Members",
        )
    )

    # Optional member IDs (text at midpoints)
    if show_member_ids:
        mx, my, mz, mt = [], [], [], []
        for eid, (i, j, E, A) in elements.items():
            xi, yi, zi = nodes3[i]
            xj, yj, zj = nodes3[j]
            mx.append((xi + xj) / 2)
            my.append((yi + yj) / 2)
            mz.append((zi + zj) / 2)
            mt.append(str(eid))

        fig.add_trace(
            go.Scatter3d(
                x=mx,
                y=my,
                z=mz,
                mode="text",
                text=mt,
                name="Member IDs",
            )
        )

    # =========================================================
    # Nodes
    # =========================================================
    nx = [p[0] for p in nodes3.values()]
    ny = [p[1] for p in nodes3.values()]
    nz = [p[2] for p in nodes3.values()]

    fig.add_trace(
        go.Scatter3d(
            x=nx,
            y=ny,
            z=nz,
            mode="markers",
            marker=dict(size=4),
            name="Nodes",
        )
    )

    if show_node_ids:
        fig.add_trace(
            go.Scatter3d(
                x=nx,
                y=ny,
                z=nz,
                mode="text",
                text=[str(nid) for nid in nodes3.keys()],
                textposition="top center",
                name="Node IDs",
            )
        )

    # =========================================================
    # Supports
    # =========================================================
    if nodes_restrained:
        sx, sy, sz = [], [], []
        for nid in nodes_restrained.keys():
            x, y, z = nodes3[nid]
            sx.append(x)
            sy.append(y)
            sz.append(z)

        fig.add_trace(
            go.Scatter3d(
                x=sx,
                y=sy,
                z=sz,
                mode="markers",
                marker=dict(size=7),
                name="Supports",
            )
        )

    # =========================================================
    # Loads (true arrows using cones)
    # =========================================================
    if nodes_loaded:
        lx, ly, lz = [], [], []
        u, v, w = [], [], []

        for nid, (Fx, Fy, Fz) in nodes_loaded.items():
            x, y, z = nodes3[nid]
            lx.append(x)
            ly.append(y)
            lz.append(z)

            # direction + magnitude
            u.append(load_scale * Fx)
            v.append(load_scale * Fy)
            w.append(load_scale * Fz)

        fig.add_trace(
            go.Cone(
                x=lx,
                y=ly,
                z=lz,
                u=u,
                v=v,
                w=w,
                anchor="tail",   # arrow starts at node
                sizemode="absolute",
                sizeref=0.2,     # adjust arrowhead size
                showscale=False,
                name="Loads",
            )
        )


    # =========================================================
    # Layout (mirrors Matplotlib intent)
    # =========================================================
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",  # equal-ish aspect
            camera=dict(
                eye=dict(x=1.4, y=1.4, z=0.9)  # similar to elev=20, azim=30
            ),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title="3D Truss",
        showlegend=True,
    )

    fig.show()





def print_dsm_results(
    u_global,
    f_global_complete,
    dof_restrained_1based,
    dof_fictitious_1based=None,
    disp_in_mm=False,
    dec=4,
    rad_dec=6,
):
    """
    3D version of the original print_dsm_results.
    """
    ndof = len(u_global)
    rows = []

    dof_restrained_1based = np.atleast_1d(dof_restrained_1based)

    if dof_fictitious_1based is None:
        dof_fictitious_1based = np.array([], dtype=int)
    else:
        dof_fictitious_1based = np.atleast_1d(dof_fictitious_1based)

    restrained_set = {int(d) for d in dof_restrained_1based}
    fictitious_set = {int(d) for d in dof_fictitious_1based}

    dof_labels = ["u_x", "u_y", "u_z", "theta_x", "theta_y", "theta_z"]
    translational_idx = {0, 1, 2}

    dof_per_node = 6

    for i in range(ndof):
        dof_1based = i + 1
        mod = i % dof_per_node
        dof_type = dof_labels[mod]

        if mod in translational_idx:
            disp = u_global[i] * (1000 if disp_in_mm else 1)
            disp_str = f"{disp:.{dec}f}"
        else:
            disp_str = f"{u_global[i]:.{rad_dec}f}"

        load_str = f"{f_global_complete[i]:.{dec}f}"

        if dof_1based in fictitious_set:
            status = "Fictitious"
        elif dof_1based in restrained_set:
            status = "Fixed"
        else:
            status = "Free"

        rows.append([dof_1based, dof_type, status, disp_str, load_str])

    disp_unit = "mm" if disp_in_mm else "m"

    df = pd.DataFrame(
        rows,
        columns=[
            "DOF",
            "Type",
            "Status",
            f"Disp ({disp_unit} / rad)",
            "Load (kN / kN·m)",
        ],
    )

    print(df.to_string(index=False))


def print_element_frame3d(
    e, u_global, m_1based, T, k_local, Qf_local, disp_in_mm=False, dec=4, rad_dec=6
):
    idx = np.asarray(m_1based, dtype=int) - 1
    u_e_global = u_global[idx]
    v_local = T @ u_e_global
    q_local = k_local @ v_local - Qf_local

    scale = 1000 if disp_in_mm else 1
    unit = "mm" if disp_in_mm else "m"

    u_out = u_e_global.copy()
    v_out = v_local.copy()

    for j in [0, 1, 2, 6, 7, 8]:
        u_out[j] *= scale
        v_out[j] *= scale

    def fmt_disp(vec):
        parts = []
        for j, val in enumerate(vec):
            if j in [3, 4, 5, 9, 10, 11]:
                parts.append(f"{val:.{rad_dec}f}")
            else:
                parts.append(f"{val:.{dec}f}")
        return "[" + ", ".join(parts) + "]"

    def fmt_force(vec):
        return "[" + ", ".join(f"{val:.{dec}f}" for val in vec) + "]"

    print(f"\nE{e} (Frame 3D)")
    print(f"u_global [{unit},rad]: {fmt_disp(u_out)}")
    print(f"v_local  [{unit},rad]: {fmt_disp(v_out)}")
    print(f"q_local  [kN,kN·m]: {fmt_force(q_local)}")


def print_element_truss3d(
    e, u_global, m_1based, T, k_local, Qf_local=None, disp_in_mm=False, dec=4
):
    idx = np.asarray(m_1based, dtype=int) - 1
    u_e_global = u_global[idx]

    if Qf_local is None:
        Qf_local = np.zeros(12, dtype=float)

    v_local = T @ u_e_global
    q_local = k_local @ v_local - Qf_local

    scale = 1000 if disp_in_mm else 1
    unit = "mm" if disp_in_mm else "m"

    u_out = u_e_global.copy()
    v_out = v_local.copy()

    for j in [0, 1, 2, 6, 7, 8]:
        u_out[j] *= scale
        v_out[j] *= scale

    def fmt(vec):
        return "[" + ", ".join(f"{val:.{dec}f}" for val in vec) + "]"

    N = q_local[6]

    print(f"\nE{e} (Truss 3D)")
    print(f"u_global [{unit},rad]: {fmt(u_out)}")
    print(f"v_local  [{unit},rad]: {fmt(v_out)}")
    print(f"q_local  [kN]: {fmt(q_local)}")
    print(f"N (tension +) = {N:.{dec}f} kN")


def print_matrix_scaled(K, scale=1000, decimals=1, col_width=3):
    fmt = f"{{:{col_width}.{decimals}f}}"
    print(f"K = {scale:.0e} ×")
    for i, row in enumerate(K, start=1):
        row_scaled = row / scale
        row_str = " ".join(fmt.format(val) for val in row_scaled)
        print(f"{i:02d} | {row_str}")


def plot_structure_3d(nodes, elements, u_global=None, scale=1.0, show_node_ids=True):
    """
    Plot undeformed and optionally deformed 3D structure.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")

    for e_id, e_data in elements.items():
        i, j = e_data["nodes"]

        xi, yi, zi = nodes[i]
        xj, yj, zj = nodes[j]

        # undeformed
        ax.plot([xi, xj], [yi, yj], [zi, zj], "k-", lw=1.8)

        # deformed
        if u_global is not None:
            ui = u_global[6 * (i - 1) : 6 * (i - 1) + 3]
            uj = u_global[6 * (j - 1) : 6 * (j - 1) + 3]

            ax.plot(
                [xi + scale * ui[0], xj + scale * uj[0]],
                [yi + scale * ui[1], yj + scale * uj[1]],
                [zi + scale * ui[2], zj + scale * uj[2]],
                "r-",
                lw=1.8,
            )

    if show_node_ids:
        for node_id, (x, y, z) in nodes.items():
            ax.text(x, y, z, f"{node_id}", fontsize=8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"3D Structure (black = undeformed, red = deformed x{scale})")
    plt.tight_layout()
    plt.show()


def _frame3d_hermite_shape_functions(xi, L):
    """
    Cubic Hermite shape functions for Euler-Bernoulli beam bending.
    xi : normalized coordinate in [0, 1]
    L  : element length
    """
    H1 = 1.0 - 3.0 * xi**2 + 2.0 * xi**3
    H2 = L * (xi - 2.0 * xi**2 + xi**3)
    H3 = 3.0 * xi**2 - 2.0 * xi**3
    H4 = L * (-xi**2 + xi**3)
    return H1, H2, H3, H4


def interpolate_frame3d_centerline_local(v_local, L, npts=41, scale=1.0):
    """
    Interpolate deformed centerline of a 3D frame element in LOCAL coordinates.

    Local DOF order:
    [ux_i, uy_i, uz_i, rx_i, ry_i, rz_i, ux_j, uy_j, uz_j, rx_j, ry_j, rz_j]

    Returns
    -------
    xyz_local : (npts, 3) ndarray
        Deformed centerline points in local coordinates.
    """
    ux_i, uy_i, uz_i, rx_i, ry_i, rz_i, ux_j, uy_j, uz_j, rx_j, ry_j, rz_j = v_local

    xi_vals = np.linspace(0.0, 1.0, npts)
    xyz_local = np.zeros((npts, 3), dtype=float)

    for k, xi in enumerate(xi_vals):
        x = xi * L

        # axial interpolation
        u = (1.0 - xi) * ux_i + xi * ux_j

        # bending in local y direction <-> rotation rz
        H1, H2, H3, H4 = _frame3d_hermite_shape_functions(xi, L)
        v = H1 * uy_i + H2 * rz_i + H3 * uy_j + H4 * rz_j

        # bending in local z direction <-> rotation ry
        # sign convention matches the current ky block in your frame stiffness
        w = H1 * uz_i - H2 * ry_i + H3 * uz_j - H4 * ry_j

        xyz_local[k, 0] = x + scale * u
        xyz_local[k, 1] = scale * v
        xyz_local[k, 2] = scale * w

    return xyz_local


def plot_structure_3d_shape_functions(
    nodes,
    elements,
    u_global,
    T_list,
    map_list,
    meta_list,
    scale=100.0,
    npts=41,
    save_path=None,
):
    """
    Plot undeformed and deformed 3D structure.
    - Truss elements: straight deformed line
    - Frame elements: deformed curve using 3D frame shape functions
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")

    # ----------------------------
    # 1) undeformed structure
    # ----------------------------
    for e_id, e_data in elements.items():
        i, j = e_data["nodes"]
        xi, yi, zi = nodes[i]
        xj, yj, zj = nodes[j]
        ax.plot([xi, xj], [yi, yj], [zi, zj], "k-", lw=1.5)

    # ----------------------------
    # 2) deformed structure
    # ----------------------------
    for meta, T, m_1based in zip(meta_list, T_list, map_list):
        etype = meta["type"]
        i, j = meta["nodes"]

        xyz_i = np.asarray(nodes[i], dtype=float)

        idx = np.asarray(m_1based, dtype=int) - 1
        u_e_global = u_global[idx]
        v_local = T @ u_e_global

        # T maps global -> local, so R is global -> local
        R = T[0:3, 0:3]

        if etype == "truss":
            ui = u_e_global[0:3]
            uj = u_e_global[6:9]

            p_i = xyz_i + scale * ui
            p_j = np.asarray(nodes[j], dtype=float) + scale * uj

            ax.plot(
                [p_i[0], p_j[0]],
                [p_i[1], p_j[1]],
                [p_i[2], p_j[2]],
                "r-",
                lw=1.8,
            )

        elif etype == "frame":
            L = meta["L"]

            xyz_local_curve = interpolate_frame3d_centerline_local(
                v_local, L, npts=npts, scale=scale
            )

            xyz_global_curve = np.zeros_like(xyz_local_curve)
            for a in range(npts):
                xyz_global_curve[a, :] = xyz_i + R.T @ xyz_local_curve[a, :]

            ax.plot(
                xyz_global_curve[:, 0],
                xyz_global_curve[:, 1],
                xyz_global_curve[:, 2],
                "r-",
                lw=1.8,
            )

        else:
            raise ValueError(f"Unknown element type: {etype}")


    # ----------------------------
    # 4) labels
    # ----------------------------
    ax.set_xlabel("X (longitudinal)")
    ax.set_ylabel("Y (vertical)")
    ax.set_zlabel("Z (transverse)")
    ax.set_title(f"3D Structure with Shape Functions (black = undeformed, red = deformed x{scale})")

    # ----------------------------
    # 5) 自动设置坐标范围（关键）
    # ----------------------------
    xs = [coord[0] for coord in nodes.values()]
    ys = [coord[1] for coord in nodes.values()]
    zs = [coord[2] for coord in nodes.values()]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)

    # 加一点边界（否则贴边很难看）
    pad_x = 0.05 * (xmax - xmin)
    pad_y = 0.15 * (ymax - ymin)   # Y方向多给点空间（竖向）
    pad_z = 0.10 * (zmax - zmin)

    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_zlim(zmin - pad_z, zmax + pad_z)

    # ----------------------------
    # 6) 纵向压缩（桥梁专用显示）
    # ----------------------------
    # X 很长（128m），Y/Z 很小 → 必须压缩比例
    ax.set_box_aspect((1, 0.3, 0.3))  
    # 如果还觉得太扁，可以改成 (1, 0.4, 0.4)

    # ----------------------------
    # 7) 更适合桥的视角
    # ----------------------------
    ax.view_init(elev=90, azim=-90)

    # ----------------------------
    # 8) 提高可读性
    # ----------------------------
    ax.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return fig


def plot_structure_3d_shape_functions_interactive(
    nodes,
    elements,
    u_global,
    T_list,
    map_list,
    meta_list,
    scale=1.0,
    npts=41,
    show_node_ids=True,
):
    """
    Interactive 3D plot using Plotly.
    - Undeformed structure: black
    - Deformed structure: red
    - Truss: straight line
    - Frame: shape-function curve
    """

    fig = go.Figure()

    # ---------------------------------
    # 1) undeformed structure
    # ---------------------------------
    for e_id, e_data in elements.items():
        i, j = e_data["nodes"]
        xi, yi, zi = nodes[i]
        xj, yj, zj = nodes[j]

        fig.add_trace(
            go.Scatter3d(
                x=[xi, xj],
                y=[yi, yj],  # keeps style consistent
                z=[zi, zj],
                mode="lines",
                line=dict(color="black", width=4),
                name=f"Undeformed E{e_id}",
                showlegend=False,
                hoverinfo="text",
                text=[f"E{e_id}", f"E{e_id}"],
            )
        )

    # ---------------------------------
    # 2) deformed structure
    # ---------------------------------
    for meta, T, m_1based in zip(meta_list, T_list, map_list):
        e_id = meta["e_id"]
        etype = meta["type"]
        i, j = meta["nodes"]

        xyz_i = np.asarray(nodes[i], dtype=float)
        xyz_j = np.asarray(nodes[j], dtype=float)

        idx = np.asarray(m_1based, dtype=int) - 1
        u_e_global = u_global[idx]
        v_local = T @ u_e_global

        R = T[0:3, 0:3]

        if etype == "truss":
            ui = u_e_global[0:3]
            uj = u_e_global[6:9]

            p_i = xyz_i + scale * ui
            p_j = xyz_j + scale * uj

            fig.add_trace(
                go.Scatter3d(
                    x=[p_i[0], p_j[0]],
                    y=[p_i[1], p_j[1]],
                    z=[p_i[2], p_j[2]],
                    mode="lines",
                    line=dict(color="red", width=5),
                    name=f"Deformed E{e_id}",
                    showlegend=False,
                    hoverinfo="text",
                    text=[f"E{e_id} (truss)", f"E{e_id} (truss)"],
                )
            )

        elif etype == "frame":
            L = meta["L"]

            xyz_local_curve = interpolate_frame3d_centerline_local(
                v_local, L, npts=npts, scale=scale
            )

            xyz_global_curve = np.zeros_like(xyz_local_curve)
            for a in range(npts):
                xyz_global_curve[a, :] = xyz_i + R.T @ xyz_local_curve[a, :]

            fig.add_trace(
                go.Scatter3d(
                    x=xyz_global_curve[:, 0],
                    y=xyz_global_curve[:, 1],
                    z=xyz_global_curve[:, 2],
                    mode="lines",
                    line=dict(color="red", width=5),
                    name=f"Deformed E{e_id}",
                    showlegend=False,
                    hoverinfo="text",
                    text=[f"E{e_id} (frame)"] * len(xyz_global_curve),
                )
            )

        else:
            raise ValueError(f"Unknown element type: {etype}")

    # ---------------------------------
    # 3) node ids
    # ---------------------------------
    if show_node_ids:
        node_ids = sorted(nodes.keys())
        xs = [nodes[n][0] for n in node_ids]
        ys = [nodes[n][1] for n in node_ids]
        zs = [nodes[n][2] for n in node_ids]
        labels = [str(n) for n in node_ids]

        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="text",
                text=labels,
                textposition="top center",
                showlegend=False,
                hoverinfo="skip",
            )
        )
    
    # ---------------------------------
    # 4) layout
    # ---------------------------------
    fig.update_layout(
        title=f"3D Structure with Shape Functions (black = undeformed, red = deformed x{scale})",
        scene=dict(
            xaxis_title="X (longitudinal)",
            yaxis_title="Y (vertical)",
            zaxis_title="Z (transverse)",
            aspectmode="data",  # keeps geometry proportions
            camera=dict(
                eye=dict(x=1.8, y=-1.8, z=0.8)
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig.show()
    return fig


def write_results_txt(
    out_path,
    u_global,
    f_global_complete,
    dof_restrained_1based,
    meta_list,
    k_list,
    T_list,
    Qf_list,
    map_list,
    disp_in_mm=False,
    dec=4,
    rad_dec=6,
):
    """
    Write summary results to a txt file.

    Contents:
    1) Global nodal DOF results
    2) Element local results
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    restrained_set = {int(d) for d in np.atleast_1d(dof_restrained_1based)}
    ndof = len(u_global)
    dof_labels = ["u_x", "u_y", "u_z", "theta_x", "theta_y", "theta_z"]
    translational_idx = {0, 1, 2}
    scale = 1000 if disp_in_mm else 1.0
    unit = "mm" if disp_in_mm else "m"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== GLOBAL RESULTS ===\n")
        f.write(
            f"{'DOF':>4} {'Type':>8} {'Status':>8} "
            f"{('Disp (' + unit + ' / rad)'):>18} {'Load (kN / kN·m)':>18}\n"
        )

        for i in range(ndof):
            dof_1based = i + 1
            mod = i % 6
            dof_type = dof_labels[mod]
            status = "Fixed" if dof_1based in restrained_set else "Free"

            if mod in translational_idx:
                disp_val = u_global[i] * scale
                disp_str = f"{disp_val:.{dec}f}"
            else:
                disp_str = f"{u_global[i]:.{rad_dec}f}"

            load_str = f"{f_global_complete[i]:.{dec}f}"

            f.write(
                f"{dof_1based:>4} {dof_type:>8} {status:>8} "
                f"{disp_str:>18} {load_str:>18}\n"
            )

        f.write("\n=== ELEMENT RESULTS ===\n")

        for meta, k_local, T, Qf_local, m_1based in zip(
            meta_list, k_list, T_list, Qf_list, map_list
        ):
            e_id = meta["e_id"]
            etype = meta["type"]

            idx = np.asarray(m_1based, dtype=int) - 1
            u_e_global = u_global[idx]
            v_local = T @ u_e_global
            q_local = k_local @ v_local - Qf_local

            u_out = u_e_global.copy()
            v_out = v_local.copy()
            for j in [0, 1, 2, 6, 7, 8]:
                u_out[j] *= scale
                v_out[j] *= scale

            def fmt_disp(vec):
                parts = []
                for j, val in enumerate(vec):
                    if j in [3, 4, 5, 9, 10, 11]:
                        parts.append(f"{val:.{rad_dec}f}")
                    else:
                        parts.append(f"{val:.{dec}f}")
                return "[" + ", ".join(parts) + "]"

            def fmt_force(vec):
                return "[" + ", ".join(f"{val:.{dec}f}" for val in vec) + "]"

            title = "Frame 3D" if etype == "frame" else "Truss 3D"
            f.write(f"\nE{e_id} ({title})\n")
            f.write(f"u_global [{unit},rad]: {fmt_disp(u_out)}\n")
            f.write(f"v_local  [{unit},rad]: {fmt_disp(v_out)}\n")
            f.write(f"q_local  [kN,kN·m]: {fmt_force(q_local)}\n")