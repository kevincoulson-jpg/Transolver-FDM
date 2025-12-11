import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import trunc_normal_

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


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
        self.mlp_new = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, act=act)
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
        Input data should be a PyTorch Geometric Data object or similar with attribute `.x`
        containing features of shape [N, space_dim + fun_dim] where N is the number of nodes.
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

    def forward(self, data):
        x, fx, T = data.x, None, None
        x = x[None, :, :]
        fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]

        for block in self.blocks:
            fx = block(fx)

        return fx[0]

if __name__ == '__main__':
    def check_mesh_size_scaling():
        # Test different mesh sizes
        test_sizes = [100, 1000, 5000, 10000]
        
        # Create model once
        model = Transolver(space_dim=3, fun_dim=2, n_layers=5, n_hidden=256, dropout=0, 
                           n_head=8, act='gelu', mlp_ratio=1, out_dim=1, slice_num=32)
        
        print("Testing Transolver with different mesh sizes:")
        print("-" * 50)
        
        for N in test_sizes:
            # Generate dummy input data for N points
            x_coords = torch.randn(N, 1) * 2
            y_coords = torch.randn(N, 1) * 2  
            z_coords = torch.randn(N, 1) * 2
            T_values = torch.randn(N, 1) * 10
            bc_flags = torch.randint(0, 2, (N, 1)).float()
            
            input_data = torch.cat([x_coords, y_coords, z_coords, T_values, bc_flags], dim=1)
            
            # Create data object
            class SimpleData:
                def __init__(self, x):
                    self.x = x
            
            data = SimpleData(input_data)
            
            # Test the model
            with torch.no_grad():
                output = model(data)
                
            print(f"N={N:5d}: Input {input_data.shape} → Output {output.shape}")
        
        print("-" * 50)
        print("✅ Model successfully handles variable mesh sizes!")
        print("💡 Key insight: Fixed slice tokens (32) make attention O(1) w.r.t. mesh size")
    
    check_mesh_size_scaling()

    import numpy as np

    def evaluate_transvolver_on_fdm_dataset(
        generator=None, 
        dataset_path="fdm_dataset.pkl", 
        metadata_path="fdm_metadata.json",
        n_train=1000, 
        n_test=200,
        n_epochs=10,
        batch_size=64,
        lr=1e-3,
        quiet=False,
    ):
        """
        Loads FDM dataset, runs samples through Transolver, and shows l2 error over epochs.
        Args:
            generator: Optionally supply an FDMDatasetGenerator instance, otherwise will load from disk.
            dataset_path: Path to .pkl dataset file.
            metadata_path: Path to .json metadata file.
            n_train: number of training samples
            n_test: number of test samples
            n_epochs: number of training epochs
            batch_size: batch size for training
            lr: learning rate for optimizer
            quiet: print less if True
        """
        import torch
        import matplotlib.pyplot as plt

        # (1) Load or get dataset
        if generator is None:
            from fdm_create_dataset import FDMDatasetGenerator
            generator = FDMDatasetGenerator(dataset_path=dataset_path, metadata_path=metadata_path)

        # flatten for easier access
        X = np.array(generator.dataset['inputs'])   # (num_samples, N, input_dim)
        Y = np.array(generator.dataset['outputs'])  # (num_samples, N, output_dim)
        total_samples = X.shape[0]
        assert X.shape[0] == Y.shape[0], "Mismatch in input/output sample size"

        # (2) Random shuffle and split
        perm = np.random.permutation(total_samples)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:n_train+n_test]
        X_train = X[train_idx]
        Y_train = Y[train_idx]
        X_test = X[test_idx]
        Y_test = Y[test_idx]

        # (3) Build model
        sample_shape = X_train[0].shape  # (N, feat_dim)
        space_dim = 2  # crude heuristics, or set manually
        fun_dim = sample_shape[1] - space_dim
        n_layers = 4   # arbitrary
        n_hidden = 128
        slice_num = 32
        dropout = 0.0
        n_head = 8
        mlp_ratio = 1
        out_dim = Y_train[0].shape[-1]
        act = "gelu"
        device = torch.device("cpu") #cuda if available
        model = Transolver(
            space_dim=space_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout=dropout,
            n_head=n_head,
            act=act,
            mlp_ratio=mlp_ratio,
            fun_dim=fun_dim,
            out_dim=out_dim,
            slice_num=slice_num
        )
        model = model.to(device)

        # (4) Setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()

        # Preprocess all tensors (move to device)
        Xtr = [torch.from_numpy(arr).float().to(device) for arr in X_train]
        Ytr = [torch.from_numpy(arr).float().to(device) for arr in Y_train]
        Xte = [torch.from_numpy(arr).float().to(device) for arr in X_test]
        Yte = [torch.from_numpy(arr).float().to(device) for arr in Y_test]
        train_samples = len(Xtr)
        test_samples = len(Xte)

        l2_train_errors = []
        l2_test_errors = []
        for epoch in range(n_epochs):
            model.train()
            perm = np.random.permutation(train_samples)
            train_l2 = 0.0
            nbatch = 0
            for i in range(0, train_samples, batch_size):
                idx = perm[i:i+batch_size]
                Xb = [Xtr[j] for j in idx]
                Yb = [Ytr[j] for j in idx]
                optimizer.zero_grad()
                loss = 0.0  # for backprop (averaged MSE)
                batch_l2 = 0.0  # for logging: average L2 (sqrt(MSE))
                for xb, yb in zip(Xb, Yb):
                    # Model expects an object with attribute .x (see usage in __main__)
                    class SimpleData:
                        def __init__(self, x):
                            self.x = x
                    inp = SimpleData(xb)
                    pred = model(inp)
                    # pred: (N, out_dim), yb: (N, out_dim)
                    # MSE for optimizer:
                    l2_sample = torch.mean((pred - yb) ** 2)
                    loss = loss + l2_sample
                    # L2 (root MSE) for metric:
                    batch_l2 += torch.sqrt(l2_sample).item()
                # gradient step (normalizing loss by batch size)
                loss = loss / len(Xb)
                loss.backward()
                optimizer.step()
                # accumulate logged metric:
                train_l2 += batch_l2 / len(Xb)  # batch average L2 error
                nbatch += 1
            l2_train_errors.append(train_l2 / nbatch)

            # (5) Evaluate on test set
            model.eval()
            with torch.no_grad():
                test_l2 = 0.0
                for xb, yb in zip(Xte, Yte):
                    class SimpleData:
                        def __init__(self, x):
                            self.x = x
                    inp = SimpleData(xb)
                    pred = model(inp)
                    l2 = torch.sqrt(torch.mean((pred - yb) ** 2)).item()
                    test_l2 += l2
                test_l2 = test_l2 / test_samples
            l2_test_errors.append(test_l2)

            if not quiet:
                print(f"Epoch {epoch+1:03d} | Train L2: {l2_train_errors[-1]:.5f} | Test L2: {l2_test_errors[-1]:.5f}")

        # (6) Plot error history
        plt.figure(figsize=(8,6))
        plt.plot(l2_train_errors, label='Train L2')
        plt.plot(l2_test_errors, label='Test L2')
        plt.xlabel('Epoch')
        plt.ylabel('L2 Error')
        plt.title('Transolver L2 Error on FDM Dataset')
        plt.legend()
        plt.grid(True)
        plt.show()
        print("Final Test L2 Error:", l2_test_errors[-1])

    # Example usage (will run if you uncomment):
    evaluate_transvolver_on_fdm_dataset(n_train=500, n_test=100, n_epochs=50)

    
