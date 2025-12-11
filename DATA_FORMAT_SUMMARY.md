# Transolver Data Format Summary

This document summarizes the data format used by Transolver based on the existing examples in the codebase.

## PyTorch Geometric Data Object Structure

All datasets use `torch_geometric.data.Data` objects with the following fields:
- `pos`: Node positions (coordinates)
- `x`: Input features (node features)
- `y`: Target/output features (what to predict)
- `surf`: Boolean mask indicating surface nodes (optional)
- `edge_index`: Edge connectivity (optional, for graph-based models)

---

## 2D Unstructured Data (Airfoil-Design-AirfRANS)

**Source**: `Airfoil-Design-AirfRANS/dataset/dataset.py`

### Format:
- **`pos`**: `[N, 2]` - Node positions
  - Columns: `[x, y]`

- **`x`** (input features): `[N, 7]` - Node features
  - Columns: `[x, y, u_x, u_y, sdf, n_x, n_y]`
  - Where:
    - `x, y`: Position coordinates (same as `pos`)
    - `u_x, u_y`: Inflow velocity components (2D)
    - `sdf`: Signed distance function (distance to boundary, 1D)
    - `n_x, n_y`: Normal vector components (2D, zero for interior points)

- **`y`** (target): `[N, 3]` - Output to predict
  - Columns: `[U_x, U_y, p, nut]` (when using sampling)
  - Or: `[U_x, U_y, p, nut]` (when using mesh nodes)
  - Where:
    - `U_x, U_y`: Velocity components (2D)
    - `p`: Pressure (1D)
    - `nut`: Turbulent viscosity (1D)

- **`surf`**: `[N]` - Boolean mask
  - `True` for surface nodes, `False` for interior nodes

---

## 3D Unstructured Data (Car-Design-ShapeNetCar)

**Source**: `Car-Design-ShapeNetCar/dataset/dataset.py`

### Format:
- **`pos`**: `[N, 3]` - Node positions
  - Columns: `[x, y, z]`

- **`x`** (input features): `[N, 7]` - Node features
  - Columns: `[x, y, z, sdf, n_x, n_y, n_z]`
  - Where:
    - `x, y, z`: Position coordinates (same as `pos`)
    - `sdf`: Signed distance function (distance to boundary, 1D)
    - `n_x, n_y, n_z`: Normal vector components (3D)

- **`y`** (target): `[N, 4]` - Output to predict
  - Columns: `[v_x, v_y, v_z, p]`
  - Where:
    - `v_x, v_y, v_z`: Velocity components (3D)
    - `p`: Pressure (1D)

- **`surf`**: `[N]` - Boolean mask
  - `True` for surface nodes, `False` for exterior/interior nodes

- **`edge_index`**: `[2, E]` - Edge connectivity (optional)
  - Graph edges connecting neighboring nodes
  - Can be computed from mesh connectivity or using radius graph

---

## Structured Mesh Data (PDE-Solving-StandardBenchmark)

### 2D Structured (e.g., Navier-Stokes, Darcy, Elasticity):
- **`pos`**: `[N, 2]` - Grid positions
  - Columns: `[x, y]`
  - Usually from `np.meshgrid` or regular grid

- **`x`**: `[N, 2]` or `[N, T_in]` - Input features
  - For static problems: `[x, y]` (just positions)
  - For time-dependent: `[N, T_in]` (initial conditions over time)

- **`y`**: `[N, 1]` or `[N, T]` - Output
  - For static: `[N, 1]` (scalar field)
  - For time-dependent: `[N, T]` (field over time)

### 3D Structured:
- **`pos`**: `[N, 3]` - Grid positions
  - Columns: `[x, y, z]`

- **`x`**: Similar to 2D but with 3D positions

---

## Key Differences: 2D vs 3D Unstructured

| Field | 2D Unstructured | 3D Unstructured |
|-------|----------------|-----------------|
| `pos` | `[N, 2]` - `[x, y]` | `[N, 3]` - `[x, y, z]` |
| `x` | `[N, 7]` - `[x, y, u_x, u_y, sdf, n_x, n_y]` | `[N, 7]` - `[x, y, z, sdf, n_x, n_y, n_z]` |
| `y` | `[N, 3]` or `[N, 4]` - `[U_x, U_y, p, nut?]` | `[N, 4]` - `[v_x, v_y, v_z, p]` |

**Note**: The 2D case includes inflow velocity (`u_x, u_y`) in the input, while 3D does not. The 2D case may also include turbulent viscosity (`nut`) in the output.

---

## Common Patterns

1. **Position is always included in `x`**: The coordinates from `pos` are typically concatenated with other features in `x`.

2. **Signed Distance Function (SDF)**: Always included in `x` for unstructured meshes. Represents distance to the boundary (positive outside, negative inside).

3. **Normal vectors**: Included in `x` for unstructured meshes. Zero for interior points, actual normals for surface points.

4. **Surface mask**: The `surf` field distinguishes boundary/surface nodes from interior nodes, which is important for boundary condition handling.

5. **Edge connectivity**: For graph-based models, `edge_index` can be:
   - Computed from mesh connectivity (from VTK/PyVista data)
   - Computed using radius graph: `radius_graph(pos, r=radius, max_num_neighbors=k)`

---

## Example: Creating 3D Unstructured Data

```python
import torch
from torch_geometric.data import Data

# Example for 3D unstructured mesh
N = 1000  # number of nodes

# Positions: [N, 3]
pos = torch.randn(N, 3)  # x, y, z coordinates

# Input features: [N, 7]
# [x, y, z, sdf, n_x, n_y, n_z]
sdf = torch.randn(N, 1)  # signed distance function
normal = torch.randn(N, 3)  # normal vectors
x = torch.cat([pos, sdf, normal], dim=1)  # [N, 7]

# Target: [N, 4]
# [v_x, v_y, v_z, p]
velocity = torch.randn(N, 3)  # velocity components
pressure = torch.randn(N, 1)  # pressure
y = torch.cat([velocity, pressure], dim=1)  # [N, 4]

# Surface mask: [N]
surf = torch.zeros(N, dtype=torch.bool)  # False for interior, True for surface

# Create Data object
data = Data(pos=pos, x=x, y=y, surf=surf)
```

---

## Notes

- All tensors should be `torch.float32` (or `torch.float`)
- The `surf` field should be boolean (`torch.bool`)
- For graph-based models, you may need to compute `edge_index` from mesh connectivity or using radius graph
- Normalization is often applied to `x` and `y` during training (see normalization code in dataset files)

