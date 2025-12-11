import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from Basic_Transvolver import Transolver

class FDMDataset(Dataset):
    """Dataset class for FDM simulation data."""
    
    def __init__(self, dataset_path="fdm_dataset.pkl", train_split=0.8):
        """
        Initialize the dataset.
        
        Args:
            dataset_path: Path to the dataset file
            train_split: Fraction of data to use for training
        """
        # Load dataset
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        
        self.inputs = data['inputs']
        self.outputs = data['outputs']
        
        # Convert to tensors
        self.inputs = [torch.FloatTensor(x) for x in self.inputs]
        self.outputs = [torch.FloatTensor(y) for y in self.outputs]
        
        # Split train/test
        split_idx = int(len(self.inputs) * train_split)
        
        if hasattr(self, 'is_train') and self.is_train:
            self.inputs = self.inputs[:split_idx]
            self.outputs = self.outputs[:split_idx]
        else:
            self.inputs = self.inputs[split_idx:]
            self.outputs = self.outputs[split_idx:]
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

class FDMTrainDataset(FDMDataset):
    """Training dataset."""
    def __init__(self, dataset_path="fdm_dataset.pkl"):
        self.is_train = True
        super().__init__(dataset_path)

class FDMTestDataset(FDMDataset):
    """Test dataset."""
    def __init__(self, dataset_path="fdm_dataset.pkl"):
        self.is_train = False
        super().__init__(dataset_path)

class SimpleData:
    """Simple data wrapper for Transolver."""
    def __init__(self, x):
        self.x = x

def train_transolver(dataset_path="fdm_dataset.pkl", 
                    epochs=100, 
                    batch_size=4, 
                    learning_rate=1e-4,
                    device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train the Transolver model on FDM dataset.
    
    Args:
        dataset_path: Path to the dataset
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use for training
    """
    
    print(f"Training on device: {device}")
    
    # Create datasets
    train_dataset = FDMTrainDataset(dataset_path)
    test_dataset = FDMTestDataset(dataset_path)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = Transolver(
        space_dim=3,      # x, y, z coordinates
        fun_dim=2,        # T, bc_flag
        n_layers=5,
        n_hidden=256,
        dropout=0.1,
        n_head=8,
        act='gelu',
        mlp_ratio=1,
        out_dim=1,        # Temperature output
        slice_num=32
    ).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Training loop
    train_losses = []
    test_losses = []
    
    print("Starting training...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_inputs, batch_outputs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_inputs = batch_inputs.to(device)
            batch_outputs = batch_outputs.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass for each sample in batch
            batch_predictions = []
            for i in range(batch_inputs.size(0)):
                data = SimpleData(batch_inputs[i])
                pred = model(data)
                batch_predictions.append(pred)
            
            # Stack predictions and compute loss
            predictions = torch.stack(batch_predictions)
            loss = criterion(predictions, batch_outputs)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        test_loss = 0.0
        test_batches = 0
        
        with torch.no_grad():
            for batch_inputs, batch_outputs in test_loader:
                batch_inputs = batch_inputs.to(device)
                batch_outputs = batch_outputs.to(device)
                
                # Forward pass for each sample in batch
                batch_predictions = []
                for i in range(batch_inputs.size(0)):
                    data = SimpleData(batch_inputs[i])
                    pred = model(data)
                    batch_predictions.append(pred)
                
                # Stack predictions and compute loss
                predictions = torch.stack(batch_predictions)
                loss = criterion(predictions, batch_outputs)
                
                test_loss += loss.item()
                test_batches += 1
        
        avg_test_loss = test_loss / test_batches
        test_losses.append(avg_test_loss)
        
        # Update scheduler
        scheduler.step(avg_test_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")
    
    # Save model
    torch.save(model.state_dict(), 'transolver_fdm_model.pth')
    print("Model saved as 'transolver_fdm_model.pth'")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return model, train_losses, test_losses

def test_model_prediction(model_path='transolver_fdm_model.pth', 
                         dataset_path="fdm_dataset.pkl",
                         device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Test the trained model on a few examples.
    
    Args:
        model_path: Path to the trained model
        dataset_path: Path to the dataset
        device: Device to use
    """
    # Load model
    model = Transolver(
        space_dim=3, fun_dim=2, n_layers=5, n_hidden=256, dropout=0.1,
        n_head=8, act='gelu', mlp_ratio=1, out_dim=1, slice_num=32
    ).to(device)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load test dataset
    test_dataset = FDMTestDataset(dataset_path)
    
    # Test on a few examples
    with torch.no_grad():
        for i in range(min(5, len(test_dataset))):
            input_data, target = test_dataset[i]
            input_data = input_data.to(device)
            target = target.to(device)
            
            # Make prediction
            data = SimpleData(input_data)
            prediction = model(data)
            
            # Calculate metrics
            mse = nn.MSELoss()(prediction, target)
            mae = nn.L1Loss()(prediction, target)
            
            print(f"Sample {i+1}:")
            print(f"  Input shape: {input_data.shape}")
            print(f"  Target shape: {target.shape}")
            print(f"  Prediction shape: {prediction.shape}")
            print(f"  MSE: {mse.item():.6f}")
            print(f"  MAE: {mae.item():.6f}")
            print(f"  Target range: [{target.min():.2f}, {target.max():.2f}]")
            print(f"  Prediction range: [{prediction.min():.2f}, {prediction.max():.2f}]")
            print()

if __name__ == "__main__":
    # Check if dataset exists
    import os
    if not os.path.exists("fdm_dataset.pkl"):
        print("Dataset not found! Please run fdm_create_dataset.py first.")
        exit()
    
    # Train the model
    model, train_losses, test_losses = train_transolver(
        dataset_path="fdm_dataset.pkl",
        epochs=50,
        batch_size=4,
        learning_rate=1e-4
    )
    
    # Test the model
    test_model_prediction() 