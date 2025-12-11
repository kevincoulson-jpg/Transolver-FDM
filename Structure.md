# Transolver Model Structure - Step-by-Step Breakdown

This document provides a detailed breakdown of the Transolver forward pass, tracing through each `nn.Module` in the entire stack.

## Overview

**Input**: `data.x` → Shape: `[N, 5]` (where N=1000, features: x,y,z,T,bc_flag)  
**Output**: `[N, 1]` (temperature values)

---

## Transolver.forward() - Main Entry Point

### Step 1: Input Preprocessing
```python
x = x[None, :, :]  # Add batch dimension: [N, 5] → [1, N, 5]
```

### Step 2: self.preprocess (MLP)
```python
fx = self.preprocess(x)  # [1, N, 5] → [1, N, 256]
```

**Inside MLP.forward():**
- **2.1**: `self.linear_pre` → `nn.Sequential(nn.Linear(5, 512), nn.GELU())`
  - `nn.Linear(5, 512)`: [1, N, 5] → [1, N, 512]
  - `nn.GELU()`: [1, N, 512] → [1, N, 512]
- **2.2**: `self.linear_post` → `nn.Linear(512, 256)`
  - [1, N, 512] → [1, N, 256]

### Step 3: Add Placeholder
```python
fx = fx + self.placeholder[None, None, :]  # [1, N, 256] + [1, 1, 256] → [1, N, 256]
```

---

## Step 4: Loop Through Transolver_blocks (5 layers)

For each `block` in `self.blocks`:

### Transolver_block.forward()
```python
fx = self.Attn(self.ln_1(fx)) + fx  # Attention + residual
fx = self.mlp(self.ln_2(fx)) + fx   # MLP + residual
```

#### Sub-step 4.1: First Attention Path
- **4.1.1**: `self.ln_1` → `nn.LayerNorm(256)`
  - [1, N, 256] → [1, N, 256]
- **4.1.2**: `self.Attn` → `Physics_Attention_Simple`

#### Inside Physics_Attention_Simple.forward()

**4.1.2.1**: `self.in_project_x` → `nn.Linear(256, 512)` (inner_dim = 8×64)
- [1, N, 256] → [1, N, 512]
- Reshape + permute: [1, N, 512] → [1, N, 8, 64] → [1, 8, N, 64]

**4.1.2.2**: `self.in_project_slice` → `nn.Linear(64, 32)`
- [1, 8, N, 64] → [1, 8, N, 32]

**4.1.2.3**: `self.softmax` → `nn.Softmax(dim=-1)`
- [1, 8, N, 32] → [1, 8, N, 32] (slice_weights)

**4.1.2.4**: Create slice_token via einsum
- [1, 8, N, 64] × [1, 8, N, 32] → [1, 8, 32, 64]

**4.1.2.5**: Self-attention on slice tokens
- `self.to_q` → `nn.Linear(64, 64, bias=False)`
  - [1, 8, 32, 64] → [1, 8, 32, 64]
- `self.to_k` → `nn.Linear(64, 64, bias=False)`
  - [1, 8, 32, 64] → [1, 8, 32, 64]
- `self.to_v` → `nn.Linear(64, 64, bias=False)`
  - [1, 8, 32, 64] → [1, 8, 32, 64]

**4.1.2.6**: Attention computation
- `self.softmax` → `nn.Softmax(dim=-1)`
- `self.dropout` → `nn.Dropout(0)`

**4.1.2.7**: Deslice back to original space
- Einsum: [1, 8, 32, 64] × [1, 8, N, 32] → [1, 8, N, 64]
- Rearrange: [1, 8, N, 64] → [1, N, 512]

**4.1.2.8**: `self.to_out` → `nn.Sequential`
- `nn.Linear(512, 256)`: [1, N, 512] → [1, N, 256]
- `nn.Dropout(0)`: [1, N, 256] → [1, N, 256]

**4.1.3**: Residual connection
- [1, N, 256] + [1, N, 256] → [1, N, 256]

#### Sub-step 4.2: Second MLP Path
- **4.2.1**: `self.ln_2` → `nn.LayerNorm(256)`
  - [1, N, 256] → [1, N, 256]
- **4.2.2**: `self.mlp` → `MLP`

#### Inside MLP.forward()
- **4.2.2.1**: `self.linear_pre` → `nn.Sequential(nn.Linear(256, 256), nn.GELU())`
  - `nn.Linear(256, 256)`: [1, N, 256] → [1, N, 256]
  - `nn.GELU()`: [1, N, 256] → [1, N, 256]
- **4.2.2.2**: `self.linear_post` → `nn.Linear(256, 256)`
  - [1, N, 256] → [1, N, 256]

**4.2.3**: Residual connection
- [1, N, 256] + [1, N, 256] → [1, N, 256]

---

## Step 5: Final Layer Processing

For the **last layer** (layer 5), after attention and MLP:

### Final Output Projection
- **5.1**: `self.ln_3` → `nn.LayerNorm(256)`
  - [1, N, 256] → [1, N, 256]
- **5.2**: `self.mlp2` → `nn.Linear(256, 1)`
  - [1, N, 256] → [1, N, 1]

---

## Step 6: Return Final Output
```python
return fx[0]  # Remove batch dimension: [1, N, 1] → [N, 1]
```

**Final Output Shape**: `[N, 1]` where N=1000 (temperature values)

---

## Summary of nn.Module Stack

1. **MLP.preprocess**: Linear(5→512) + GELU + Linear(512→256)
2. **5× Transolver_block**, each containing:
   - **LayerNorm** → **Physics_Attention_Simple** → **LayerNorm** → **MLP**
3. **Physics_Attention_Simple**: 4× Linear projections + Softmax + Dropout
4. **Final layer**: Additional LayerNorm + Linear(256→1)

This transforms your mesh data `[N, 5]` → `[N, 1]` through a sophisticated attention mechanism!

## Key Components

### Physics_Attention_Simple
- **Purpose**: Implements a sliced attention mechanism for irregular meshes
- **Key Operations**: 
  - Projects input to multi-head format
  - Creates slice tokens via weighted aggregation
  - Applies self-attention on slice tokens
  - Projects back to original space

### MLP
- **Purpose**: Feed-forward network with residual connections
- **Structure**: Linear → Activation → Linear with skip connection

### Transolver_block
- **Purpose**: Transformer encoder block with attention and MLP
- **Structure**: LayerNorm → Attention → Residual → LayerNorm → MLP → Residual

---

## Variable Mesh Size Support

The Transolver model is designed to handle **variable N** (different numbers of mesh points) without any code changes.

### Key Design Feature: Fixed Slice Tokens

The model uses a clever "slicing" mechanism in `Physics_Attention_Simple`:

```python
# Input: [1, 8, N, 64] (variable N)
slice_weights = self.softmax(self.in_project_slice(x_mid))  # [1, 8, N, 32]
slice_token = torch.einsum("bhnc,bhng->bhgc", x_mid, slice_weights)  # [1, 8, 32, 64]
```

- **Variable input**: `[1, 8, N, 64]` where N can be any number
- **Fixed slice tokens**: `[1, 8, 32, 64]` - always 32 tokens regardless of N
- **Self-attention**: Works on 32 tokens (O(32²) complexity)
- **Deslice back**: `[1, 8, N, 64]` - back to variable N

### Computational Complexity

- **Attention operations**: O(slice_num²) = O(32²) = **constant** (independent of N)
- **Linear projections**: O(N) = **linear scaling** with mesh size
- **Memory usage**: O(N) = **linear scaling**

### Advantages

1. **Scalability**: Can handle meshes from 100 to 10,000+ points
2. **Efficiency**: Attention complexity is constant w.r.t. mesh size
3. **Flexibility**: Same model works for different mesh densities
4. **Permutation invariance**: Order of mesh points doesn't matter

### Example Usage

```python
# Same model handles different mesh sizes
model = Transolver(space_dim=3, fun_dim=2, n_layers=5, n_hidden=256, 
                  dropout=0, n_head=8, act='gelu', mlp_ratio=1, 
                  out_dim=1, slice_num=32)

# Works for any N
N = 1000    # Input: [1000, 5] → Output: [1000, 1]
N = 5000    # Input: [5000, 5] → Output: [5000, 1]
N = 10000   # Input: [10000, 5] → Output: [10000, 1]
```

This design makes the model particularly suitable for scientific computing applications where mesh sizes can vary significantly between problems. 