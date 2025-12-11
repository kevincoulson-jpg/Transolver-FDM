import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from timm.layers import trunc_normal_

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU,'softplus': nn.Softplus,}

class Physics_Attention_Simple(nn.Module):
    """
    Physics-informed attention mechanism with slice-based token aggregation.
    
    This attention module uses a slice-based approach to reduce computational complexity
    from O(N²) to O(N) by aggregating nodes into a fixed number of slice tokens.
    The attention is performed among slice tokens rather than individual nodes.
    
    Args:
        dim (int): Input feature dimension.
        heads (int, optional): Number of attention heads. Defaults to 8.
        dim_head (int, optional): Dimension per attention head. Defaults to 64.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        slice_num (int, optional): Number of slice tokens to aggregate nodes into. Defaults to 64.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape

        ### (1) Slice
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", x_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)

class MLP(nn.Module):
    """
    Multi-layer perceptron with residual connections.
    
    A feedforward network with an initial projection layer, optional hidden layers
    with residual connections, and a final output projection layer.
    
    Args:
        n_input (int): Input feature dimension.
        n_hidden (int): Hidden layer dimension.
        n_output (int): Output feature dimension.
        n_layers (int, optional): Number of hidden layers with residual connections. Defaults to 1.
        act (str, optional): Activation function name. Must be one of: 'gelu', 'tanh', 
            'sigmoid', 'relu', 'leaky_relu', 'softplus', 'ELU', 'silu'. Defaults to 'gelu'.
    
    Raises:
        NotImplementedError: If the specified activation function is not supported.
    """
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu'):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            x = self.linears[i](x) + x
        x = self.linear_post(x)
        return x

class Transolver_block(nn.Module):
    """
    Transformer encoder block for Transolver architecture.
    
    A standard transformer block with physics-informed attention, layer normalization,
    and feedforward MLP with residual connections. The last layer can optionally
    project to the output dimension.
    
    Args:
        num_heads (int): Number of attention heads.
        hidden_dim (int): Hidden feature dimension.
        dropout (float): Dropout probability.
        act (str, optional): Activation function name. Defaults to 'gelu'.
        mlp_ratio (int, optional): Ratio of MLP hidden dimension to input dimension. Defaults to 4.
        last_layer (bool, optional): Whether this is the final block that outputs predictions. Defaults to False.
        out_dim (int, optional): Output dimension (only used if last_layer=True). Defaults to 1.
        slice_num (int, optional): Number of slice tokens for attention mechanism. Defaults to 32.
    """

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Simple(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                     dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx

class Transolver(nn.Module):
    """
    Transolver: Transformer-based solver for physics-informed learning on unstructured meshes.
    
    A neural network architecture designed for solving PDEs on unstructured meshes with
    variable node counts. Uses physics-informed attention with slice-based token aggregation
    to achieve O(N) complexity instead of O(N²) for standard attention.
    
    Args:
        space_dim (int, optional): Spatial dimension (2 for 2D, 3 for 3D). Defaults to 1.
        n_layers (int, optional): Number of transformer blocks. Defaults to 5.
        n_hidden (int, optional): Hidden feature dimension. Defaults to 256.
        dropout (float, optional): Dropout probability. Defaults to 0.
        n_head (int, optional): Number of attention heads. Defaults to 8.
        act (str, optional): Activation function name. Defaults to 'gelu'.
        mlp_ratio (int, optional): Ratio of MLP hidden dimension to input dimension. Defaults to 1.
        fun_dim (int, optional): Additional feature dimension beyond spatial coordinates. Defaults to 1.
        out_dim (int, optional): Output dimension (number of fields to predict). Defaults to 1.
        slice_num (int, optional): Number of slice tokens for attention mechanism. Defaults to 32.
    
    Note:
        Input data should be a tensor of shape [N, space_dim + fun_dim] where N is the number of nodes.
        For backward compatibility, objects with a `.x` attribute are also supported.
    """
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0,
                 n_head=8,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=1,
                 out_dim=1,
                 slice_num=32
                 ):
        super(Transolver, self).__init__()
        self.__name__ = 'Transolver'

        self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, act=act)

        self.n_hidden = n_hidden
        self.space_dim = space_dim

        self.blocks = nn.ModuleList([Transolver_block(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio, out_dim=out_dim,
                                                      slice_num=slice_num,
                                                      last_layer=(_ == n_layers - 1))
                                     for _ in range(n_layers)])
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Forward pass of the Transolver model.
        
        Args:
            x: Input tensor of shape [N, space_dim + fun_dim] or [B, N, space_dim + fun_dim]
            where B is batch size, N is number of nodes.
            Can also be a PyTorch Geometric Data object with .x attribute (for backward compatibility).
        
        Returns:
            Output tensor of shape [N, out_dim] or [B, N, out_dim] (same batch dimension as input)
        """
        # Handle both tensor input and object with .x attribute (backward compatibility)
        if hasattr(x, 'x'):
            x = x.x
        
        # Ensure x is 3D: [B, N, features]
        if x.dim() == 2:
            x = x[None, :, :]  # Add batch dimension: [1, N, features]
            single_sample = True
        elif x.dim() == 3:
            single_sample = False
        else:
            raise ValueError(f"Input x must be 2D [N, features] or 3D [B, N, features], got shape {x.shape}")
        
        fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]

        for block in self.blocks:
            fx = block(fx)

        # Return with same batch structure as input
        if single_sample:
            return fx[0]  # Remove batch dimension: [N, out_dim] for single sample
        else:
            return fx  # Keep batch dimension: [B, N, out_dim] for batches
