import torch
import torch.nn as nn
from Physics_Attention import (
    Physics_Attention_Irregular_Mesh,
    Physics_Attention_Structured_Mesh_2D,
    Physics_Attention_Structured_Mesh_3D
)

def test_irregular_mesh():
    """Test the irregular mesh attention module"""
    print("=== Testing Physics_Attention_Irregular_Mesh ===")
    
    # Model parameters
    dim = 128
    heads = 8
    dim_head = 64
    slice_num = 32
    
    # Create model (untrained)
    model = Physics_Attention_Irregular_Mesh(
        dim=dim, 
        heads=heads, 
        dim_head=dim_head, 
        slice_num=slice_num
    )
    
    # Create dummy input
    batch_size = 2
    N = 100  # number of points
    C = dim  # feature dimension
    
    x = torch.randn(batch_size, N, C)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: {x.shape}")
    print(f"Shape matches: {output.shape == x.shape}")
    
    return model, x, output

def test_structured_mesh_2d():
    """Test the 2D structured mesh attention module"""
    print("\n=== Testing Physics_Attention_Structured_Mesh_2D ===")
    
    # Model parameters
    dim = 128
    heads = 8
    dim_head = 64
    slice_num = 32
    H, W = 20, 15  # grid dimensions
    
    # Create model (untrained)
    model = Physics_Attention_Structured_Mesh_2D(
        dim=dim, 
        heads=heads, 
        dim_head=dim_head, 
        slice_num=slice_num,
        H=H, W=W
    )
    
    # Create dummy input
    batch_size = 2
    N = H * W  # total grid points
    C = dim
    
    x = torch.randn(batch_size, N, C)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: {x.shape}")
    print(f"Shape matches: {output.shape == x.shape}")
    
    return model, x, output

def test_structured_mesh_3d():
    """Test the 3D structured mesh attention module"""
    print("\n=== Testing Physics_Attention_Structured_Mesh_3D ===")
    
    # Model parameters
    dim = 128
    heads = 8
    dim_head = 64
    slice_num = 16
    H, W, D = 10, 8, 6  # grid dimensions
    
    # Create model (untrained)
    model = Physics_Attention_Structured_Mesh_3D(
        dim=dim, 
        heads=heads, 
        dim_head=dim_head, 
        slice_num=slice_num,
        H=H, W=W, D=D
    )
    
    # Create dummy input
    batch_size = 2
    N = H * W * D  # total grid points
    C = dim
    
    x = torch.randn(batch_size, N, C)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: {x.shape}")
    print(f"Shape matches: {output.shape == x.shape}")
    
    return model, x, output

def visualize_with_torchviz(model, x, output, model_name):
    """Visualize the model using torchviz"""
    try:
        from torchviz import make_dot
        print(f"\n=== Generating torchviz graph for {model_name} ===")
        
        # Create the computation graph
        dot = make_dot(output, params=dict(model.named_parameters()))
        
        # Render the graph
        filename = f"{model_name.lower().replace(' ', '_')}_graph"
        dot.render(filename, format="png", cleanup=True)
        print(f"Graph saved as {filename}.png")
        
    except ImportError:
        print("torchviz not installed. Install with: pip install torchviz")

def visualize_with_tensorboard(model, x, model_name):
    """Visualize the model using TensorBoard"""
    try:
        from torch.utils.tensorboard import SummaryWriter
        print(f"\n=== Generating TensorBoard graph for {model_name} ===")
        
        # Create writer
        writer = SummaryWriter(f"runs/{model_name.lower().replace(' ', '_')}")
        
        # Add graph
        writer.add_graph(model, x)
        writer.close()
        
        print(f"TensorBoard log saved. Run: tensorboard --logdir=runs")
        
    except ImportError:
        print("tensorboard not installed. Install with: pip install tensorboard")

if __name__ == "__main__":
    # Test all models
    models_data = []
    
    # Test irregular mesh
    model1, x1, output1 = test_irregular_mesh()
    models_data.append(("Physics_Attention_Irregular_Mesh", model1, x1, output1))
    
    # Test 2D structured mesh
    model2, x2, output2 = test_structured_mesh_2d()
    models_data.append(("Physics_Attention_Structured_Mesh_2D", model2, x2, output2))
    
    # Test 3D structured mesh
    model3, x3, output3 = test_structured_mesh_3d()
    models_data.append(("Physics_Attention_Structured_Mesh_3D", model3, x3, output3))
    
    # Generate visualizations
    for model_name, model, x, output in models_data:
        # visualize_with_torchviz(model, x, output, model_name)
        visualize_with_tensorboard(model, x, model_name)
    
    print("\n=== Validation Complete ===")
    print("All models should maintain input/output shape consistency.")
    print("Check the generated PNG files and TensorBoard logs for visualization.") 