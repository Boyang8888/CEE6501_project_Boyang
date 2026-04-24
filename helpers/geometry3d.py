
import numpy as np


def element_xyzL(xyz_i, xyz_j):
    """
    Return:
        ex : unit vector along local x-axis
        L  : element length
    """
    xyz_i = np.asarray(xyz_i, dtype=float)
    xyz_j = np.asarray(xyz_j, dtype=float)

    dxyz = xyz_j - xyz_i
    L = np.linalg.norm(dxyz)

    if L <= 0.0:
        raise ValueError("Zero-length element detected.")

    ex = dxyz / L
    return ex, float(L)


def _reference_vector_from_ex(ex):
    """
    Pick a stable reference vector not parallel to ex.
    """
    global_z = np.array([0.0, 0.0, 1.0])
    global_y = np.array([0.0, 1.0, 0.0])

    if abs(np.dot(ex, global_z)) < 0.90:
        return global_z
    return global_y


def rotation_matrix_3d(xyz_i, xyz_j, reference_vector=None):
    """
    Build a 3x3 direction-cosine matrix R.

    Local axes:
    local x = along element
    local y = as close as possible to global Y (vertical)
    local z = completes right-handed system

    This makes qy in member_loads approximately vertical.
    """
    ex, _ = element_xyzL(xyz_i, xyz_j)

    global_y = np.array([0.0, 1.0, 0.0], dtype=float)
    global_z = np.array([0.0, 0.0, 1.0], dtype=float)

    # Prefer global Y as local y reference
    ref = global_y.copy()

    # If element is nearly vertical, use global Z instead
    if abs(np.dot(ex, ref)) > 0.90:
        ref = global_z.copy()

    # Remove component of ref along local x
    ey = ref - np.dot(ref, ex) * ex
    norm_ey = np.linalg.norm(ey)

    if norm_ey < 1e-12:
        raise ValueError("Could not construct stable local y-axis.")

    ey = ey / norm_ey

    # Complete right-handed coordinate system
    ez = np.cross(ex, ey)
    ez = ez / np.linalg.norm(ez)

    R = np.vstack([ex, ey, ez])
    return R


def transformation_matrix_12x12(R):
    """
    Build the 12x12 transformation matrix for a 3D frame/truss element.
    """
    T = np.zeros((12, 12), dtype=float)

    T[0:3, 0:3] = R
    T[3:6, 3:6] = R
    T[6:9, 6:9] = R
    T[9:12, 9:12] = R

    return T


def build_elements_RL(elements, nodes, planar_frame_xy=False):
    """
    Return:
        elements_RL[e_id] = {"R": R, "L": L}
    """
    elements_RL = {}

    for e_id, e_data in elements.items():
        i, j = e_data["nodes"]

        xyz_i = nodes[i]
        xyz_j = nodes[j]

        if planar_frame_xy and e_data.get("type", "") == "frame":
            R = rotation_matrix_3d_planar_xy(xyz_i, xyz_j)
        else:
            R = rotation_matrix_3d(xyz_i, xyz_j)
        _, L = element_xyzL(xyz_i, xyz_j)

        elements_RL[e_id] = {
            "R": R,
            "L": L,
        }

    return elements_RL


def rotation_matrix_3d_planar_xy(xyz_i, xyz_j):
    """
    Local axes for a 2D frame embedded in the global x-y plane.

    local x: along the element
    local z: fixed to global Z
    local y: z x x  (right-hand rule)
    """
    xi = np.asarray(xyz_i, dtype=float)
    xj = np.asarray(xyz_j, dtype=float)

    dx = xj - xi
    L = np.linalg.norm(dx)
    ex = dx / L

    ez = np.array([0.0, 0.0, 1.0], dtype=float)   # fixed global Z
    ey = np.cross(ez, ex)
    ey = ey / np.linalg.norm(ey)

    # re-orthogonalize z
    ez = np.cross(ex, ey)
    ez = ez / np.linalg.norm(ez)

    R = np.vstack([ex, ey, ez])
    return R