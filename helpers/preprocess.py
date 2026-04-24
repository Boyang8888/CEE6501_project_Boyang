
import json
import numpy as np

DOF_LABELS_3D = ["ux", "uy", "uz", "rx", "ry", "rz"]







def read_model_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _int_key_dict(d):
    if d is None:
        return {}
    out = {}
    for k, v in d.items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            out[k] = v
    return out


def normalize_model_data(data):
    """
    Normalize top-level JSON fields into Python-friendly dictionaries.

    Expected JSON fields
    --------------------
    nodes
    materials
    sections
    elements
    nodes_restrained
    nodal_loads
    nodes_prescribed  (optional)
    """
    model = {}
    model["nodes"] = _int_key_dict(data.get("nodes", {}))
    model["materials"] = data.get("materials", {})
    model["sections"] = data.get("sections", {})
    model["elements"] = _int_key_dict(data.get("elements", {}))
    model["nodes_restrained"] = _int_key_dict(data.get("nodes_restrained", {}))
    model["nodal_loads"] = _int_key_dict(data.get("nodal_loads", {}))
    model["nodes_prescribed"] = _int_key_dict(data.get("nodes_prescribed", {}))
    model["member_loads"] = parse_member_loads(data)  
    model["releases"] = _int_key_dict(data.get("releases", {}))
    model["temperature_loads"] = data.get("temperature_loads", {})
    model["fabrication_errors"] = _int_key_dict(data.get("fabrication_errors", {}))
    return model


def infer_element_type(section_name):
    """
    First-version project rule:
    A1, A3 -> truss
    A2, A4 -> frame
    """
    if section_name in ["A1", "A3"]:
        return "truss"
    if section_name in ["A2", "A4"]:
        return "frame"
    raise ValueError(f"Unknown section {section_name}. Cannot infer element type.")


def parse_elements(elements_raw):
    """
    Convert raw element definitions into a unified dictionary format.

    Supported input forms
    ---------------------
    1) "1": [i, j, "A1"]
    2) "1": {"nodes":[i,j], "section":"A1", "type":"truss"}

    Output format
    -------------
    elements[element_id] = {
        "nodes": [i, j],
        "section": "A1",
        "type": "truss" or "frame"
    }
    """
    elements = {}

    for e_id, e_data in elements_raw.items():
        if isinstance(e_data, dict):
            i, j = e_data["nodes"]
            sec = e_data["section"]
            etype = e_data.get("type", infer_element_type(sec))
        else:
            i, j, sec = e_data
            etype = infer_element_type(sec)

        elements[int(e_id)] = {
            "nodes": [int(i), int(j)],
            "section": sec,
            "type": etype,
        }

    return elements


def get_single_material(materials):
    """
    First-version simplification:
    use one material for all elements.
    """
    if not materials:
        raise ValueError("No materials found in the input JSON.")

    if len(materials) == 1:
        return next(iter(materials.values()))

    if "steel" in materials:
        return materials["steel"]

    return next(iter(materials.values()))


def build_global_nodal_load_vector(nodes, nodal_loads):
    """
    Build the 6N global nodal load vector from:
    nodal_loads[node_id] = [Fx, Fy, Fz, Mx, My, Mz]
    """
    ndof = 6 * len(nodes)
    F_global = np.zeros(ndof, dtype=float)

    for node_id, load_vals in nodal_loads.items():
        if len(load_vals) != 6:
            raise ValueError(
                f"Node {node_id} load must have 6 components [Fx,Fy,Fz,Mx,My,Mz]."
            )
        start = 6 * (int(node_id) - 1)
        F_global[start : start + 6] += np.asarray(load_vals, dtype=float)

    return F_global


def build_restrained_dofs_1based(nodes_restrained):
    """
    Convert:
        nodes_restrained = {1: ["ux","uy"], 5:["uz"]}
    into a sorted 1-based DOF list.
    """
    dof_lookup = {
        "ux": 1,
        "uy": 2,
        "uz": 3,
        "rx": 4,
        "ry": 5,
        "rz": 6,
    }

    dof_restrained_1based = []

    for node_id, labels in nodes_restrained.items():
        for label in labels:
            if label not in dof_lookup:
                raise ValueError(f"Unknown DOF label {label} at node {node_id}.")
            dof_1based = 6 * (int(node_id) - 1) + dof_lookup[label]
            dof_restrained_1based.append(dof_1based)

    return np.array(sorted(set(dof_restrained_1based)), dtype=int)


def build_prescribed_displacement_vector(nodes, nodes_prescribed):
    """
    Optional support settlements / prescribed displacements.

    Format
    ------
    nodes_prescribed = {
        1: {"ux": 0.0, "uy": 0.0},
        5: {"uy": -0.002}
    }

    Returns
    -------
    u_global : full global displacement vector with prescribed values filled,
               all others zero.
    """
    ndof = 6 * len(nodes)
    u_global = np.zeros(ndof, dtype=float)

    dof_lookup = {
        "ux": 0,
        "uy": 1,
        "uz": 2,
        "rx": 3,
        "ry": 4,
        "rz": 5,
    }

    for node_id, dof_dict in nodes_prescribed.items():
        base = 6 * (int(node_id) - 1)
        for label, value in dof_dict.items():
            if label not in dof_lookup:
                raise ValueError(f"Unknown prescribed DOF label {label} at node {node_id}.")
            u_global[base + dof_lookup[label]] = float(value)

    return u_global



def element_lmnL(xy_i, xy_j):
    xy_i = np.asarray(xy_i, dtype=float)
    xy_j = np.asarray(xy_j, dtype=float)

    dx = xy_j[0] - xy_i[0]
    dy = xy_j[1] - xy_i[1]
    dz = xy_j[2] - xy_i[2]
    L = float(np.sqrt(dx**2 + dy**2 + dz**2))

    l = dx / L
    m = dy / L
    n = dz / L
    e_x = np.array([l, m, n])

    if abs(n) < 0.9:
        a = np.array([0, 0, 1])
    else:
        a = np.array([0, 1, 0])

    e_y = np.cross(a, e_x) / np.linalg.norm(np.cross(a, e_x))
    e_z = np.cross(e_x, e_y)

    l_y, m_y, n_y = e_y
    l_z, m_z, n_z = e_z

    return float(l), float(m), float(n), float(l_y), float(m_y), float(n_y), float(l_z), float(m_z), float(n_z), float(L)


def parse_member_loads(data):
    raw = data.get("member_loads", {})
    member_loads = {}

    for e_id_str, load_list in raw.items():
        e_id = int(e_id_str)
        member_loads[e_id] = []

        for item in load_list:
            load_type = item["type"]

            if load_type == "udl_local":
                member_loads[e_id].append({
                    "type": "udl_local",
                    "qx": float(item.get("qx", 0.0)),
                    "qy": float(item.get("qy", 0.0)),
                    "qz": float(item.get("qz", 0.0)),
                })

            elif load_type == "point_local":
                member_loads[e_id].append({
                    "type": "point_local",
                    "Px": float(item.get("Px", 0.0)),
                    "Py": float(item.get("Py", 0.0)),
                    "Pz": float(item.get("Pz", 0.0)),
                    "a": float(item["a"]),
                })

            else:
                raise ValueError(f"Unsupported member load type: {load_type}")

    return member_loads