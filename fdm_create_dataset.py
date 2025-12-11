import numpy as np
import pickle
import hashlib
import os
from datetime import datetime
import json
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from fdm_laser import FDM_Laser, HeatSourceLaser, make_path

class FDMDatasetGenerator:
    def __init__(self, dataset_path="fdm_dataset.pkl", metadata_path="fdm_metadata.json"):
        """
        Initialize the dataset generator.
        
        Args:
            dataset_path: Path to save the dataset
            metadata_path: Path to save simulation metadata
        """
        self.dataset_path = dataset_path
        self.metadata_path = metadata_path
        self.dataset = {'inputs': [], 'outputs': [], 'sim_hashes': []}
        self.metadata = {'simulations': [], 'total_samples': 0, 'created': str(datetime.now())}
        
        # Load existing dataset if it exists
        self.load_dataset()
    
    def load_dataset(self):
        """Load existing dataset and metadata if they exist."""
        if os.path.exists(self.dataset_path):
            with open(self.dataset_path, 'rb') as f:
                self.dataset = pickle.load(f)
            print(f"Loaded existing dataset with {len(self.dataset['inputs'])} samples")
        
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"Loaded metadata with {len(self.metadata['simulations'])} simulations")
    
    def save_dataset(self):
        """Save dataset and metadata to disk."""
        with open(self.dataset_path, 'wb') as f:
            pickle.dump(self.dataset, f)
        
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Saved dataset with {len(self.dataset['inputs'])} samples")
    
    def generate_simulation_hash(self, params):
        """Generate unique hash for simulation parameters."""
        # Convert parameters to string and hash
        param_str = str(sorted(params.items()))
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def extract_training_data(self, save_dict):
        """
        Extract training data from FDM simulation results.
        
        The feature extraction is now done during simulation in FDM_Laser.solve(),
        so this method simply returns the already-processed data.
        
        Uses every time step pair from the simulation:
        - Input: feature vector at time step i-1 [x, y, k, rho, C, temperature, I, bc]
        - Output: temperature at time step i
        
        Args:
            save_dict: Dictionary with 'input' and 'output' arrays
                - 'input': List of (N*N, 8) feature arrays
                - 'output': List of (N*N, 1) temperature arrays
                Each entry corresponds to one time step pair (i-1 -> i)
            
        Returns:
            inputs: List of (N*N, 8) arrays [x, y, k, rho, C, temperature, I, bc]
            outputs: List of (N*N, 1) arrays [temperature_next]
        """
        # Data is already processed during simulation, just return it
        inputs = save_dict['input']
        outputs = save_dict['output']
        
        return inputs, outputs
    
    def run_simulation(self, params):
        """
        Run a single FDM simulation with given parameters.
        
        Args:
            params: Dictionary of simulation parameters
            
        Returns:
            inputs, outputs: Training data from simulation
        """
        # Create laser based on parameters
        if params['laser_type'] == 'static':
            laser = HeatSourceLaser(dr=params['laser_dr'], position=params['laser_center'])
        elif params['laser_type'] == 'circle':
            # Generate path with enough points for smooth interpolation
            # Estimate n_path based on path length for smoothness
            radius = params.get('circle_radius', params['l']/4)
            path_length = 2 * np.pi * radius
            n_path = max(64, int(path_length / 0.001))  # ~1mm spacing
            path_points = make_path(
                mode='circle', l=params['l'], n_path=n_path,
                center=params['laser_center'], radius=params['circle_radius']
            )
            laser = HeatSourceLaser(dr=params['laser_dr'], path=path_points, 
                                  velocity=params['laser_velocity'])
        elif params['laser_type'] == 'rectangle':
            # Estimate n_path based on perimeter
            width = params.get('path_width', params['l']/2)
            height = params.get('path_height', params['l']/2)
            path_length = 2 * (width + height)
            n_path = max(64, int(path_length / 0.001))  # ~1mm spacing
            path_points = make_path(
                mode='rectangle', l=params['l'], n_path=n_path,
                center=params['laser_center'], width=params['path_width'],
                height=params['path_height']
            )
            laser = HeatSourceLaser(dr=params['laser_dr'], path=path_points,
                                  velocity=params['laser_velocity'])
        elif params['laser_type'] == 'snake':
            # Estimate n_path based on snaking area
            width = params.get('path_width', params['l']/2)
            height = params.get('path_height', params['l']/2)
            spacing = params.get('path_spacing', width/10)
            n_rows = max(2, int(height / spacing) + 1)
            path_length = n_rows * width
            n_path = max(64, int(path_length / 0.001))  # ~1mm spacing
            path_points = make_path(
                mode='snake', l=params['l'], n_path=n_path,
                center=params['laser_center'], width=params['path_width'],
                height=params['path_height'], spacing=params['path_spacing']
            )
            laser = HeatSourceLaser(dr=params['laser_dr'], path=path_points,
                                  velocity=params['laser_velocity'])
        else:
            raise ValueError(f"Unknown laser type: {params['laser_type']}")
        
        # Create save dictionary
        save_dict = {'input': [], 'output': []}
        
        # Create FDM instance
        fdm = FDM_Laser(
            N=params['N'], l=params['l'], dt=params['dt'], tf=params['tf'],
            K=params['K'], rho=params['rho'], C=params['C'],
            theta0=params['theta0'], theta_w=params['theta_w'],
            a=params['a'], I0=params['I0'], laser=laser,
            write_output=True, save_dict=save_dict
        )
        
        # Run simulation
        print(f"Running simulation with hash: {self.generate_simulation_hash(params)[:8]}...")
        try:
            final_temp = fdm.solve(update_fn='matmul')
            print(f"Simulation completed. Max temp: {np.max(final_temp):.2f}K")
            
            # Extract training data
            inputs, outputs = self.extract_training_data(save_dict)
            return inputs, outputs
            
        except Exception as e:
            print(f"Simulation failed: {e}")
            return [], []
    
    def generate_parameter_combinations(self, param_ranges, num_simulations):
        """
        Generate random parameter combinations for simulation runs.
        
        Args:
            param_ranges: Dictionary with parameter ranges
                - 'base': Fixed parameters for all simulations
                - 'variable': Dictionary of variable parameters
                    - Numeric parameters: [min, max] (will be randomly sampled)
                    - Non-numeric parameters: list of possible values (randomly chosen)
                    - Special: 'laser_center_x' and 'laser_center_y' are composed into
                      'laser_center' tuple (x, y) automatically
            num_simulations: Number of random parameter combinations to generate
            
        Returns:
            List of parameter dictionaries with 'laser_center' as (x, y) tuple
        """
        # Base parameters (fixed)
        base_params = param_ranges.get('base', {})
        
        # Variable parameters
        var_params = param_ranges.get('variable', {})
        
        param_combinations = []
        
        for _ in range(num_simulations):
            params = base_params.copy()
            
            # Randomly sample each variable parameter
            for key, value_range in var_params.items():
                # Skip laser_center_x and laser_center_y - we'll handle them specially
                if key in ['laser_center_x', 'laser_center_y']:
                    continue
                    
                # Check if it's a numeric range [min, max]
                if (isinstance(value_range, list) and len(value_range) == 2 and
                    isinstance(value_range[0], (int, float)) and 
                    isinstance(value_range[1], (int, float))):
                    # Random value in range
                    params[key] = np.random.uniform(value_range[0], value_range[1])
                else:
                    # List of possible values (non-numeric or discrete choices)
                    params[key] = np.random.choice(value_range)
            
            # Compose laser_center from laser_center_x and laser_center_y if they exist
            if 'laser_center_x' in var_params and 'laser_center_y' in var_params:
                # Sample x
                x_range = var_params['laser_center_x']
                if (isinstance(x_range, list) and len(x_range) == 2 and
                    isinstance(x_range[0], (int, float)) and isinstance(x_range[1], (int, float))):
                    center_x = np.random.uniform(x_range[0], x_range[1])
                else:
                    center_x = np.random.choice(x_range)
                
                # Sample y
                y_range = var_params['laser_center_y']
                if (isinstance(y_range, list) and len(y_range) == 2 and
                    isinstance(y_range[0], (int, float)) and isinstance(y_range[1], (int, float))):
                    center_y = np.random.uniform(y_range[0], y_range[1])
                else:
                    center_y = np.random.choice(y_range)
                
                # Compose into tuple
                params['laser_center'] = (center_x, center_y)
            
            param_combinations.append(params)
        
        return param_combinations
    
    def add_simulations(self, param_ranges, num_simulations, max_samples_per_sim=None):
        """
        Add new simulations to the dataset.
        
        Args:
            param_ranges: Dictionary defining parameter ranges
                - 'base': Fixed parameters for all simulations
                - 'variable': Dictionary of variable parameters
                    - Numeric parameters: [min, max] (will be randomly sampled)
                    - Non-numeric parameters: list of possible values (randomly chosen)
                    - Special: 'laser_center_x' and 'laser_center_y' are composed into
                      'laser_center' tuple (x, y) automatically
            num_simulations: Number of random parameter combinations to generate and run
            max_samples_per_sim: Maximum samples to use per simulation
        """
        param_combinations = self.generate_parameter_combinations(param_ranges, num_simulations)
        
        new_simulations = 0
        skipped_simulations = 0
        total_samples_added = 0
        total_combinations = len(param_combinations)
        
        for idx, params in enumerate(param_combinations, 1):
            sim_hash = self.generate_simulation_hash(params)
            
            # Check if simulation already exists
            if sim_hash in self.dataset['sim_hashes']:
                skipped_simulations += 1
                if skipped_simulations <= 3 or (idx % 100 == 0):
                    print(f"  [{idx}/{total_combinations}] Simulation {sim_hash[:8]} already exists, skipping...")
                continue
            
            # Show progress
            print(f"\n  [{idx}/{total_combinations}] Running simulation {sim_hash[:8]}...")
            print(f"      Parameters: I0={params['I0']:.2e}, type={params['laser_type']}, "
                  f"dr={params['laser_dr']:.4f}, center={params['laser_center']}")
            
            # Run simulation
            inputs, outputs = self.run_simulation(params)
            
            if len(inputs) > 0:
                original_count = len(inputs)
                # Limit samples if specified
                if max_samples_per_sim and len(inputs) > max_samples_per_sim:
                    # Randomly sample to limit size
                    indices = np.random.choice(len(inputs), max_samples_per_sim, replace=False)
                    inputs = [inputs[i] for i in indices]
                    outputs = [outputs[i] for i in indices]
                    print(f"      Sampled {len(inputs)}/{original_count} time steps")
                
                # Add to dataset
                self.dataset['inputs'].extend(inputs)
                self.dataset['outputs'].extend(outputs)
                self.dataset['sim_hashes'].extend([sim_hash] * len(inputs))
                
                # Update metadata
                self.metadata['simulations'].append({
                    'hash': sim_hash,
                    'params': params,
                    'samples': len(inputs),
                    'timestamp': str(datetime.now())
                })
                
                new_simulations += 1
                total_samples_added += len(inputs)
                
                print(f"      ✓ Added {len(inputs)} samples (total: {total_samples_added:,})")
            else:
                print(f"      ✗ Simulation failed or produced no data")
        
        self.metadata['total_samples'] = len(self.dataset['inputs'])
        
        print(f"\n   Dataset update complete:")
        print(f"     - Processed: {total_combinations} parameter combinations")
        print(f"     - New simulations: {new_simulations}")
        print(f"     - Skipped (already exist): {skipped_simulations}")
        print(f"     - New samples added: {total_samples_added:,}")
        print(f"     - Total samples in dataset: {self.metadata['total_samples']:,}")
        
        # Save dataset
        self.save_dataset()
    
    def get_dataset_stats(self):
        """Print dataset statistics."""
        print(f"\nDataset Statistics:")
        print(f"- Total samples: {len(self.dataset['inputs'])}")
        print(f"- Total simulations: {len(self.metadata['simulations'])}")
        print(f"- Dataset created: {self.metadata['created']}")
        
        if len(self.dataset['inputs']) > 0:
            sample_shape = self.dataset['inputs'][0].shape
            print(f"- Sample input shape: {sample_shape}")
            print(f"- Sample output shape: {self.dataset['outputs'][0].shape}")
    
    def shuffle_dataset(self):
        """Shuffle the dataset randomly."""
        if len(self.dataset['inputs']) == 0:
            return
        
        indices = np.random.permutation(len(self.dataset['inputs']))
        
        self.dataset['inputs'] = [self.dataset['inputs'][i] for i in indices]
        self.dataset['outputs'] = [self.dataset['outputs'][i] for i in indices]
        self.dataset['sim_hashes'] = [self.dataset['sim_hashes'][i] for i in indices]
        
        print("Dataset shuffled!")
    
    def visualize_dataset(self):
        """
        Interactive visualization tool for the dataset.
        
        Displays a random sample from the dataset showing all input features
        and output in 2D. Includes a button to randomly select a new sample.
        """
        if len(self.dataset['inputs']) == 0:
            print("Dataset is empty. Cannot visualize.")
            return
        
        # Get first sample to infer grid size
        sample_input = self.dataset['inputs'][0]
        N = int(np.sqrt(sample_input.shape[0]))  # Grid size
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.08], hspace=0.3, wspace=0.3)
        
        # Create axes for plots
        axes = []
        for i in range(2):
            row = []
            for j in range(3):
                row.append(fig.add_subplot(gs[i, j]))
            axes.append(row)
        
        # Button axis
        button_ax = fig.add_subplot(gs[2, :])
        
        # Store colorbars to remove on update
        colorbars = [None] * 6
        
        def update_plot(sample_idx=None):
            """Update the plot with a new random sample."""
            # Select random sample if not specified
            if sample_idx is None:
                sample_idx = np.random.randint(0, len(self.dataset['inputs']))
            
            input_sample = self.dataset['inputs'][sample_idx]
            output_sample = self.dataset['outputs'][sample_idx]
            
            # Reshape to 2D
            x_2d = input_sample[:, 0].reshape(N, N)
            y_2d = input_sample[:, 1].reshape(N, N)
            k_2d = input_sample[:, 2].reshape(N, N)
            rho_2d = input_sample[:, 3].reshape(N, N)
            C_2d = input_sample[:, 4].reshape(N, N)
            temp_2d = input_sample[:, 5].reshape(N, N)
            I_2d = input_sample[:, 6].reshape(N, N)
            bc_2d = input_sample[:, 7].reshape(N, N)
            temp_next_2d = output_sample[:, 0].reshape(N, N)
            
            # Remove old colorbars
            for cb in colorbars:
                if cb is not None:
                    cb.remove()
            
            # Clear axes
            for ax in axes[0] + axes[1]:
                ax.clear()
            
            # Get domain size from x coordinates
            l = x_2d.max()
            
            # Plot 1: Temperature (input)
            im1 = axes[0][0].contourf(x_2d, y_2d, temp_2d, levels=50, cmap='hot')
            axes[0][0].set_title(f'Input: Temperature (K)\nSample {sample_idx+1}/{len(self.dataset["inputs"])}')
            axes[0][0].set_xlabel('x (m)')
            axes[0][0].set_ylabel('y (m)')
            colorbars[0] = plt.colorbar(im1, ax=axes[0][0])
            
            # Plot 2: Laser Intensity
            im2 = axes[0][1].contourf(x_2d, y_2d, I_2d, levels=20, cmap='Reds')
            axes[0][1].set_title(f'Input: Laser Intensity (W/m²)\nNon-zero: {np.count_nonzero(I_2d)} points')
            axes[0][1].set_xlabel('x (m)')
            axes[0][1].set_ylabel('y (m)')
            colorbars[1] = plt.colorbar(im2, ax=axes[0][1])
            
            # Plot 3: Boundary Condition
            im3 = axes[0][2].imshow(bc_2d, extent=[0, l, 0, l], origin='lower', cmap='RdYlGn', vmin=0, vmax=1)
            axes[0][2].set_title('Input: Boundary Condition\n(0=interior, 1=Dirichlet)')
            axes[0][2].set_xlabel('x (m)')
            axes[0][2].set_ylabel('y (m)')
            colorbars[2] = plt.colorbar(im3, ax=axes[0][2])
            
            # Plot 4: Temperature Next (output)
            im4 = axes[1][0].contourf(x_2d, y_2d, temp_next_2d, levels=50, cmap='hot')
            axes[1][0].set_title('Output: Temperature Next (K)')
            axes[1][0].set_xlabel('x (m)')
            axes[1][0].set_ylabel('y (m)')
            colorbars[3] = plt.colorbar(im4, ax=axes[1][0])
            
            # Plot 5: Temperature Change
            temp_diff = temp_next_2d - temp_2d
            im5 = axes[1][1].contourf(x_2d, y_2d, temp_diff, levels=50, cmap='coolwarm')
            axes[1][1].set_title('Temperature Change (Next - Current)')
            axes[1][1].set_xlabel('x (m)')
            axes[1][1].set_ylabel('y (m)')
            colorbars[4] = plt.colorbar(im5, ax=axes[1][1])
            
            # Plot 6: Material Properties (showing k, rho, C as RGB)
            # Normalize each property to [0, 1] for RGB display
            k_norm = (k_2d - k_2d.min()) / (k_2d.max() - k_2d.min() + 1e-10)
            rho_norm = (rho_2d - rho_2d.min()) / (rho_2d.max() - rho_2d.min() + 1e-10)
            C_norm = (C_2d - C_2d.min()) / (C_2d.max() - C_2d.min() + 1e-10)
            material_rgb = np.stack([k_norm, rho_norm, C_norm], axis=2)
            im6 = axes[1][2].imshow(material_rgb, extent=[0, l, 0, l], origin='lower')
            axes[1][2].set_title('Input: Material Properties\n(R=k, G=rho, B=C)')
            axes[1][2].set_xlabel('x (m)')
            axes[1][2].set_ylabel('y (m)')
            # Note: RGB images don't need colorbar
            
            # Print sample info
            print(f"\nDisplaying sample {sample_idx+1}/{len(self.dataset['inputs'])}")
            print(f"  Temperature range: [{temp_2d.min():.2f}, {temp_2d.max():.2f}] K")
            print(f"  Temperature next range: [{temp_next_2d.min():.2f}, {temp_next_2d.max():.2f}] K")
            print(f"  Laser intensity: {I_2d.max():.2e} W/m² ({np.count_nonzero(I_2d)} points)")
            print(f"  Material: k={k_2d[0,0]:.2f}, rho={rho_2d[0,0]:.0f}, C={C_2d[0,0]:.0f}")
            
            fig.canvas.draw_idle()
        
        # Create button
        button = Button(button_ax, 'Random Sample', color='lightblue', hovercolor='lightcyan')
        
        def on_button_click(event):
            """Handle button click to show random sample."""
            update_plot()
        
        button.on_clicked(on_button_click)
        
        # Initialize with first random sample
        update_plot()
        
        plt.show()

def example_usage():
    """Example of how to use the dataset generator."""
    
    print("="*70)
    print("FDM DATASET GENERATOR - EXAMPLE USAGE")
    print("="*70)
    
    # Create dataset generator
    print("\n1. Initializing dataset generator...")
    generator = FDMDatasetGenerator()
    
    # Define parameter ranges for simulations
    print("\n2. Defining parameter ranges...")
    num_simulations = 5  # Number of random parameter combinations to generate
    
    param_ranges = {
        'base': {
            # Fixed parameters
            'N': 51,
            'l': 0.2,
            'dt': 0.1,
            'tf': 50.0,  # Shorter simulation time
            'K': 35,
            'rho': 7700,
            'C': 500,
            'theta0': 300,
            'theta_w': 300,
            'a': 0.8,
        },
        'variable': {
            # Variable parameters
            # Numeric parameters: [min, max] (will be randomly sampled)
            'I0': [1e9, 5e9],  # Laser intensity range
            'laser_dr': [0.002, 0.01],  # Laser size range
            'laser_velocity': [0.001, 0.02],  # Laser velocity range (m/s)
            'circle_radius': [0.04, 0.08],  # For circle type
            'path_width': [0.08, 0.16],  # For rectangle/snake type
            'path_height': [0.06, 0.16],  # For rectangle/snake type
            'path_spacing': [0.02, 0.06],  # For snake type
            
            # Non-numeric parameters: list of possible values (randomly chosen)
            'laser_type': ['static', 'circle', 'rectangle', 'snake'],
            
            # Laser center coordinates (composed into tuple later)
            'laser_center_x': [0.09, 0.11],  # X coordinate range
            'laser_center_y': [0.09, 0.11],  # Y coordinate range
        }
    }
    
    # Print parameter ranges summary
    print("\n   Base (fixed) parameters:")
    for key, value in param_ranges['base'].items():
        print(f"     {key}: {value}")
    
    print("\n   Variable parameters:")
    for key, value_range in param_ranges['variable'].items():
        if isinstance(value_range, list) and len(value_range) == 2:
            if isinstance(value_range[0], (int, float)) and isinstance(value_range[1], (int, float)):
                print(f"     {key}: [{value_range[0]}, {value_range[1]}] (numeric range)")
            else:
                print(f"     {key}: {value_range} (list of values)")
        else:
            print(f"     {key}: {value_range} (list of values)")
    
    print(f"\n   Number of simulations to generate: {num_simulations}")
    
    # Generate parameter combinations to show examples
    print("\n3. Generating random parameter combinations...")
    param_combinations = generator.generate_parameter_combinations(param_ranges, num_simulations)
    print(f"   Generated {len(param_combinations)} random parameter combinations")
    
    # Show first few examples
    print("\n   Example parameter combinations (first 3):")
    for i, params in enumerate(param_combinations[:3]):
        print(f"\n   Combination {i+1}:")
        print(f"     I0: {params['I0']:.2e}")
        print(f"     laser_dr: {params['laser_dr']:.4f}")
        print(f"     laser_type: {params['laser_type']}")
        print(f"     laser_center: {params['laser_center']}")
        if params['laser_type'] == 'circle':
            print(f"     circle_radius: {params['circle_radius']:.4f}")
        elif params['laser_type'] in ['rectangle', 'snake']:
            print(f"     path_width: {params['path_width']:.4f}, path_height: {params['path_height']:.4f}")
            if params['laser_type'] == 'snake':
                print(f"     path_spacing: {params['path_spacing']:.4f}")
    
    if len(param_combinations) > 3:
        print(f"\n   ... and {len(param_combinations) - 3} more combinations")
    
    # Check how many already exist
    existing_count = 0
    for params in param_combinations:
        sim_hash = generator.generate_simulation_hash(params)
        if sim_hash in generator.dataset['sim_hashes']:
            existing_count += 1
    
    new_count = len(param_combinations) - existing_count
    print(f"\n   Simulations to run: {new_count} new, {existing_count} already exist")
    
    # Add simulations to dataset
    print(f"\n4. Running simulations (max {100} samples per simulation)...")
    print("   " + "-"*66)
    generator.add_simulations(param_ranges, num_simulations, max_samples_per_sim=500)
    
    # Shuffle dataset
    print("\n5. Shuffling dataset...")
    generator.shuffle_dataset()
    
    # Print statistics
    print("\n6. Final dataset statistics:")
    generator.get_dataset_stats()
    
    print("\n" + "="*70)
    print("DATASET GENERATION COMPLETE")
    print("="*70)
    
    # Optional: Visualize dataset
    print("\n7. To visualize the dataset, run:")
    print("   generator.visualize_dataset()")
    # Uncomment the line below to automatically open visualization
    generator.visualize_dataset()

if __name__ == "__main__":
    # example_usage()

    # Load and visualize existing dataset/metadata
    generator = FDMDatasetGenerator()
    generator.visualize_dataset()
