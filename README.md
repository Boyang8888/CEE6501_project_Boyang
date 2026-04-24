

# CEE6501 Matrix Structural Analysis Project

This repository contains a Python-based structural analysis program developed for the **CEE 6501 Matrix Structural Analysis** final project. The program implements the **Direct Stiffness Method (DSM)** for analyzing 2D and 3D structural systems.

---

## 📌 Project Overview

The program is designed to:

* Analyze truss and frame structures in 3D
* Support mixed truss–frame modeling
* Handle multiple engineering effects
* Perform validation against hand-calculated problems
* Study a real-world bridge-inspired structure

The final model is based on a **128 m simply supported steel truss span** inspired by the **Nanjing Yangtze River Bridge**.

---

## ⚙️ Main Features

* 3D truss element analysis
* 3D frame element analysis
* Mixed truss + frame modeling
* Global stiffness matrix assembly
* Nodal displacement and support reaction calculation
* Element local force recovery
* Prescribed support displacement (settlement)
* Uniform temperature loading
* Fabrication error (member length deviation)
* Member end release (hinge modeling)
* JSON-based input system
* Static and interactive visualization (Matplotlib + Plotly)

---

## 🐍 Python Environment

Developed and tested with:

```
Python 3.12.12
```

Required packages:

```
numpy
matplotlib
plotly
```

---

## 📂 Repository Structure

```
CEE6501_project_Boyang/
│
├── main.ipynb                 # Main entry point
│
├── inputs/                    # Input JSON files
│   ├── final_structure.json
│   ├── final_structure_no_tem.json
│   ├── validation_truss.json
│   ├── validation_frame.json
│   └── ...
│
├── outputs/                   # Output results (auto-generated)
│
└── helpers/                   # Core modules
    ├── preprocess.py
    ├── geometry3d.py
    ├── truss3d.py
    ├── frame3d.py
    ├── assembly.py
    ├── solver.py
    └── postprocess.py
```

---

## ▶️ How to Run

Open `main.ipynb` and run:

```python
main(r"C:\Users\bchen601\Documents\GitHub\CEE6501_project_Boyang\inputs\final_structure.json")
```

For the case **without temperature effects**:

```python
main(r"C:\Users\bchen601\Documents\GitHub\CEE6501_project_Boyang\inputs\final_structure_no_tem.json")
```

---

## 🧾 Input File Format

The program uses JSON to define structural models.

### Example structure:

```json
{
  "nodes": {},
  "elements": {},
  "materials": {},
  "sections": {},
  "nodes_restrained": {},
  "nodes_prescribed": {},
  "nodal_loads": {},
  "member_loads": {},
  "temperature_loads": {},
  "fabrication_errors": {},
  "releases": {}
}
```

### Key components:

* `nodes` → node coordinates
* `elements` → connectivity and element type
* `materials` → material properties
* `sections` → cross-sectional properties
* `nodes_restrained` → boundary conditions
* `nodes_prescribed` → support displacements
* `nodal_loads` → nodal forces
* `member_loads` → distributed loads
* `temperature_loads` → thermal effects
* `fabrication_errors` → length deviations
* `releases` → member end releases

---

## 📐 DOF Convention

Each node has 6 DOFs:

```
[u_x, u_y, u_z, theta_x, theta_y, theta_z]
```

Coordinate system:

```
X → longitudinal
Y → vertical
Z → transverse
```

---

## ✅ Validation Cases

The program is verified using multiple benchmark problems:

* 2D truss verification (A5 Code)
* Frame baseline case (A7 Q2)
* Support settlement (W10 Q1)
* Fabrication error (A10 Q3)
* Member release (A8)

Results show:

* Excellent agreement with hand calculations
* Correct handling of boundary conditions
* Accurate internal force distribution
* Proper implementation of releases and imperfections

---

## 🌉 Final Structure Study

The final structure includes:

* 76 nodes
* Mixed truss + frame elements
* Vertical nodal loads
* Distributed loads on floor beams
* Support settlement
* Temperature effects
* Fabrication error

### Modeling strategy:

* Truss elements → main truss members (axial-dominated)
* Frame elements → floor beams (bending-dominated)

---

## 📊 Key Results

With temperature effects:

```
Max vertical displacement:      -50.596 mm
Max longitudinal displacement:   20.892 mm
Max transverse displacement:     25.204 mm

Max axial force:                3640 kN
Max stress:                     121.347 MPa
```

### Observations:

* Vertical displacement → dominated by support settlement
* Longitudinal displacement → dominated by temperature
* Transverse displacement → reduced due to stiffness redistribution
* Internal forces → axial-force dominated (truss behavior)

---

## 📁 Output

The program generates:

* Console results
* `.txt` summary files
* Static deformation plots (`.png`)
* Interactive 3D visualization (Plotly)

All outputs are saved in:

```
outputs/
```

---

## ⚠️ Limitations

* Simplified loading assumptions (nodal + UDL)
* No real bridge design data
* Idealized connections (rigid / pinned)
* No temperature gradient
* No stochastic fabrication errors
* No semi-rigid connection modeling

👉 Results reflect **global structural behavior**, not detailed design-level accuracy.

---

## 👤 Author

**Boyang Chen**


---

## ⭐ Notes

This project demonstrates:

* Full DSM implementation
* Engineering-level modeling decisions
* Multi-effect structural analysis
* Validation against analytical solutions

---
