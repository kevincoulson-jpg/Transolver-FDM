import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.sparse import diags, eye, kron
import time

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.3f} seconds.")
        return result
    return wrapper

def make_path(mode, l, n_path=64, **kwargs):
    """
    Generate a path for the laser.
    
    Args:
        mode: 'circle', 'rectangle', or 'snake'
        l: domain size
        n_path: number of points in the path
        **kwargs: additional parameters
            - For 'circle': center (tuple), radius (float)
            - For 'rectangle': center (tuple), width (float), height (float)
            - For 'snake': center (tuple), width (float), height (float), spacing (float)
    
    Returns:
        List of (x, y) tuples representing the path points
    """
    path_points = []
    if mode == 'circle':
        center = kwargs.get('center', (l/2, l/2))
        radius = kwargs.get('radius', l/4)
        for i in range(n_path):
            angle = 2 * np.pi * i / n_path
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            path_points.append((x, y))
    elif mode == 'rectangle':
        center = kwargs.get('center', (l/2, l/2))
        width = kwargs.get('width', l/2)
        height = kwargs.get('height', l/2)
        # Rectangle corners
        x0, y0 = center[0] - width/2, center[1] - height/2
        x1, y1 = center[0] + width/2, center[1] + height/2
        n_side = n_path // 4
        # Bottom
        for i in range(n_side):
            x = x0 + (x1 - x0) * i / n_side
            y = y0
            path_points.append((x, y))
        # Right
        for i in range(n_side):
            x = x1
            y = y0 + (y1 - y0) * i / n_side
            path_points.append((x, y))
        # Top
        for i in range(n_side):
            x = x1 - (x1 - x0) * i / n_side
            y = y1
            path_points.append((x, y))
        # Left
        for i in range(n_side):
            x = x0
            y = y1 - (y1 - y0) * i / n_side
            path_points.append((x, y))
    elif mode == 'snake':
        center = kwargs.get('center', (l/2, l/2))
        width = kwargs.get('width', l/2)
        height = kwargs.get('height', l/2)
        spacing = kwargs.get('spacing', width / 10)
        
        # Calculate starting position (bottom-left of the snaking area)
        x_start = center[0] - width/2
        y_start = center[1] - height/2
        
        # Calculate number of rows based on height and spacing
        n_rows = max(2, int(height / spacing) + 1)
        # Calculate points per row to distribute n_path evenly
        n_per_row = max(2, n_path // n_rows)
        
        # Generate snaking path: left-to-right, then right-to-left, alternating
        for row in range(n_rows):
            y = y_start + row * spacing
            # Clamp y to not exceed height
            y = min(y, y_start + height)
            
            if row % 2 == 0:  # Even rows: left to right
                for i in range(n_per_row):
                    x = x_start + (width * i / (n_per_row - 1)) if n_per_row > 1 else x_start
                    x = min(x, x_start + width)  # Clamp to width
                    path_points.append((x, y))
            else:  # Odd rows: right to left
                for i in range(n_per_row):
                    x = x_start + width - (width * i / (n_per_row - 1)) if n_per_row > 1 else x_start + width
                    x = max(x, x_start)  # Clamp to width
                    path_points.append((x, y))
        
        # Trim to exactly n_path points if needed
        if len(path_points) > n_path:
            # Sample evenly
            indices = np.linspace(0, len(path_points) - 1, n_path, dtype=int)
            path_points = [path_points[i] for i in indices]
    else:
        raise ValueError("Unknown path mode")
    return path_points

class HeatSourceLaser:
    """
    Represents a moving or static laser heat source for FDM simulations.
    
    This class manages the position and movement of a laser heat source that can
    either remain static at a fixed location or follow a predefined path. The laser
    is represented as a square region with side length 2*dr centered at a position.
    
    Parameters
    ----------
    dr : float
        Half-width of the laser beam (radius in each direction from position).
        The laser covers a square region of size 2*dr x 2*dr.
    path : list of tuples, optional
        List of (x, y) coordinates defining the path the laser should follow.
        If provided, the laser will move along this path at the specified velocity.
        If None, laser is static.
    velocity : float, optional
        Velocity along the path (m/s). Required if path is provided.
        Position is updated every time step based on velocity.
    position : tuple, optional
        Initial position (x, y) for static laser. Required if path is None.
        If path is provided, position is set to the first path point.
    
    Attributes
    ----------
    position : tuple
        Current position (x, y) of the laser.
    dr : float
        Half-width of the laser beam.
    path : list
        List of path points (x, y) tuples.
    velocity : float
        Velocity along the path (m/s).
    path_length : float
        Total length of the path (calculated from path points).
    time_elapsed : float
        Total time elapsed along the path.
    
    Properties
    ----------
    xl_min, xl_max, yl_min, yl_max : float
        Bounds of the laser zone (position ± dr).
    
    Methods
    -------
    get_bounds()
        Get the bounding box of the laser zone.
    update_position(dt)
        Update position along path based on velocity and time step dt.
    visualize_path(l, N)
        Plot the laser path coverage on a grid.
    
    Examples
    --------
    >>> # Static laser at position
    >>> laser = HeatSourceLaser(dr=0.005, position=(0.1, 0.1))
    >>> 
    >>> # Moving laser following a circle path
    >>> path = make_path('circle', l=0.2, n_path=64, 
    ...                  center=(0.1, 0.1), radius=0.05)
    >>> laser = HeatSourceLaser(dr=0.005, path=path, velocity=0.1)
    """
    def __init__(self, dr, path=None, velocity=None, position=None):
        self.path = path if path is not None else []
        self.dr = dr
        self.velocity = velocity
        self.time_elapsed = 0.0
        
        if self.path:
            if velocity is None:
                raise ValueError('velocity must be provided when path is specified')
            self.position = self.path[0]
            # Calculate path length
            self.path_length = self._calculate_path_length()
        elif position is not None:
            self.position = position  # (x, y)
            self.path_length = 0.0
        else:
            raise ValueError('Must provide either a path with velocity or a position for HeatSourceLaser')
    
    def _calculate_path_length(self):
        """Calculate total length of the path."""
        if len(self.path) < 2:
            return 0.0
        total_length = 0.0
        for i in range(len(self.path)):
            p1 = self.path[i]
            p2 = self.path[(i + 1) % len(self.path)]  # Wrap around for closed paths
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            total_length += np.sqrt(dx*dx + dy*dy)
        return total_length
    
    def _get_position_at_distance(self, distance):
        """Get position along path at a given distance from start."""
        if len(self.path) < 2:
            return self.path[0] if self.path else self.position
        
        # Normalize distance to path length (for looping)
        distance = distance % self.path_length if self.path_length > 0 else 0.0
        
        # Find which segment we're on
        accumulated = 0.0
        for i in range(len(self.path)):
            p1 = self.path[i]
            p2 = self.path[(i + 1) % len(self.path)]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            segment_length = np.sqrt(dx*dx + dy*dy)
            
            if accumulated + segment_length >= distance:
                # Interpolate along this segment
                t = (distance - accumulated) / segment_length if segment_length > 0 else 0.0
                x = p1[0] + t * dx
                y = p1[1] + t * dy
                return (x, y)
            
            accumulated += segment_length
        
        # Fallback to last point
        return self.path[-1]
    @property
    def xl_min(self):
        return self.position[0] - self.dr
    @property
    def xl_max(self):
        return self.position[0] + self.dr
    @property
    def yl_min(self):
        return self.position[1] - self.dr
    @property
    def yl_max(self):
        return self.position[1] + self.dr
    def get_bounds(self):
        return self.xl_min, self.xl_max, self.yl_min, self.yl_max
    def update_position(self, dt):
        """
        Update position along path based on velocity and time step.
        
        Args:
            dt: Time step (seconds)
        """
        if not self.path:
            return  # Static laser, no update needed
        
        # Update time elapsed
        self.time_elapsed += dt
        
        # Calculate distance traveled
        distance = self.velocity * self.time_elapsed
        
        # Get position at this distance
        self.position = self._get_position_at_distance(distance)
    def visualize_path(self, l, N):
        print(f'len(self.path): {len(self.path)}')
        if len(self.path) == 0:
            print('No path provided, using single point laser')
            return
        x = np.linspace(0, l, N)
        y = np.linspace(0, l, N)
        mask = np.zeros((N, N))
        # Save current position
        original_position = self.position
        for pt in self.path:
            self.position = pt
            xl_min, xl_max, yl_min, yl_max = self.get_bounds()
            x_laser_min = round(xl_min / l * (N - 1))
            x_laser_max = round(xl_max / l * (N - 1))
            y_laser_min = round(yl_min / l * (N - 1))
            y_laser_max = round(yl_max / l * (N - 1))
            mask[x_laser_min:x_laser_max+1, y_laser_min:y_laser_max+1] = 1
        # Restore original position
        self.position = original_position
        # Plot at the end
        plt.figure(figsize=(6, 6))
        plt.imshow(mask.T, extent=[0, l, 0, l], origin='lower', cmap='gray', vmin=0, vmax=1)
        # Draw white square perimeter at the last position
        xl_min, xl_max, yl_min, yl_max = self.get_bounds()
        rect = plt.Rectangle((xl_min, yl_min), 2*self.dr, 2*self.dr, fill=False, edgecolor='white', linewidth=2)
        plt.gca().add_patch(rect)
        plt.title('Laser path coverage and last position')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.xlim(0, l)
        plt.ylim(0, l)
        plt.tight_layout()
        plt.show()

class FDM_Laser:
    """
    Finite Difference Method solver for 2D heat equation with a moving laser heat source.
    
    This class solves the 2D heat conduction equation using finite difference methods,
    with a moving laser heat source that can follow various path patterns (circle, 
    rectangle, snake, etc.). The solver uses Dirichlet boundary conditions (fixed 
    temperature at all edges) and supports both loop-based and matrix multiplication 
    based time stepping.
    
    The heat equation solved is:
        ∂θ/∂t = (K/(ρC)) * ∇²θ + (1/(ρC)) * Q(x,y,t)
    
    where Q(x,y,t) is the laser heat source term.
    
    Parameters
    ----------
    N : int
        Grid size (N x N grid points).
    l : float
        Domain size in meters (square domain: l x l).
    dt : float
        Time step size in seconds.
    tf : float
        Final simulation time in seconds.
    K : float
        Thermal conductivity (W/m·K).
    rho : float
        Material density (kg/m³).
    C : float
        Specific heat capacity (J/kg·K).
    theta0 : float
        Initial temperature throughout the domain (K).
    theta_w : float
        Wall/boundary temperature (Dirichlet BC) (K).
    a : float
        Absorption coefficient (dimensionless, 0-1).
    I0 : float
        Laser intensity (W/m²).
    laser : HeatSourceLaser
        Laser object that defines the heat source position and path.
    write_output : bool, optional
        If True, save input/output feature vectors at each time step.
        Inputs are (N*N, 8) arrays with features [x, y, k, rho, C, temperature, I, bc].
        Outputs are (N*N, 1) arrays with next temperature.
        Default is False.
    save_dict : dict, optional
        Dictionary to store input/output pairs. If None, creates a new dict.
        Keys: 'input' (list of feature arrays), 'output' (list of temperature arrays).
        Default is None.
    
    Attributes
    ----------
    x, y : ndarray
        1D arrays of spatial coordinates.
    X, Y : ndarray
        2D meshgrid arrays for plotting.
    L : sparse matrix
        2D Laplacian operator matrix.
    dx, dy : float
        Spatial step sizes.
    
    Methods
    -------
    solve(update_fn='loop')
        Solve the heat equation over the time interval [0, tf].
    plot_field(field, title, vmin=None, vmax=None)
        Plot a 2D temperature field with laser zone overlay.
    fdm_step(t_prev)
        Perform one time step using loop-based computation.
    fdm_step_matmul(t_prev)
        Perform one time step using matrix multiplication (faster).
    
    Examples
    --------
    >>> laser = HeatSourceLaser(dr=0.005, path=path_points, path_dt=0.2)
    >>> fdm = FDM_Laser(N=51, l=0.2, dt=0.01, tf=50, K=35, rho=7700, 
    ...                 C=500, theta0=300, theta_w=300, a=0.8, I0=5e9, laser=laser)
    >>> final_temp = fdm.solve(update_fn='matmul')
    """
    def __init__(self, N, l, dt, tf, K, rho, C, theta0, theta_w, a, I0, laser: HeatSourceLaser, write_output=False, save_dict=None):
        self.N = N
        self.l = l
        self.dt = dt
        self.tf = tf
        self.K = K
        self.rho = rho
        self.C = C
        self.theta0 = theta0
        self.theta_w = theta_w
        self.a = a
        self.I0 = I0
        self.laser = laser
        self.dx = l / (N - 1)
        self.dy = self.dx
        self.x = np.linspace(0, l, N)
        self.y = np.linspace(0, l, N)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.L = self.fdmA_2D(N)
        self.write_output = write_output
        self.save_dict = save_dict if save_dict is not None else {'input': [], 'output': []}

    def fdmA_2D(self, N):
        """
        Create a 2D finite difference Laplacian matrix with Dirichlet BCs.
        """
        main_diag = -2 * np.ones(N)
        off_diag = np.ones(N-1)
        A1D = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(N, N))
        I = eye(N)
        L = kron(A1D, I) + kron(I, A1D)
        return L

    def apply_dirichlet_boundaries(self, t):
        """
        Apply Dirichlet (fixed temperature) boundary conditions to all edges.
        """
        t[:, 0] = self.theta_w    # Left wall
        t[:, -1] = self.theta_w   # Right wall
        t[0, :] = self.theta_w    # Bottom wall
        t[-1, :] = self.theta_w   # Top wall

    def get_laser_indices(self):
        xl_min, xl_max, yl_min, yl_max = self.laser.get_bounds()
        x_laser_min = round(xl_min / self.l * (self.N - 1))
        x_laser_max = round(xl_max / self.l * (self.N - 1))
        y_laser_min = round(yl_min / self.l * (self.N - 1))
        y_laser_max = round(yl_max / self.l * (self.N - 1))
        return x_laser_min, x_laser_max, y_laser_min, y_laser_max
    
    def get_laser_intensity(self, i, j, laser_indices):
        """
        Get laser intensity for a grid point.
        
        Args:
            i, j: grid indices
            laser_indices: (x_min, x_max, y_min, y_max) for laser zone
            
        Returns:
            I: Laser intensity (I0 if in laser zone, 0 otherwise)
        """
        x_laser_min, x_laser_max, y_laser_min, y_laser_max = laser_indices
        
        # Check if in laser zone
        if (x_laser_min <= i <= x_laser_max and 
            y_laser_min <= j <= y_laser_max):
            return self.I0  # In laser beam
        
        return 0.0  # Not in laser beam
    
    def create_bc_flag(self, i, j):
        """
        Create boundary condition flag for a grid point.
        
        Args:
            i, j: grid indices
            
        Returns:
            bc_flag: 0 = interior point, 1 = Dirichlet boundary
        """
        # Check if on domain boundary
        if i == 0 or i == self.N-1 or j == 0 or j == self.N-1:
            return 1  # Dirichlet boundary
        
        return 0  # Interior point
    
    def extract_features(self, t_field, laser_indices):
        """
        Extract feature vectors for all grid points.
        
        Args:
            t_field: (N, N) temperature field
            laser_indices: (x_min, x_max, y_min, y_max) for laser zone
            
        Returns:
            features: (N*N, 8) array [x, y, k, rho, C, temperature, I, bc]
        """
        # Flatten 2D grid to 1D points
        x_flat = self.X.flatten()  # (N*N,)
        y_flat = self.Y.flatten()  # (N*N,)
        t_flat = t_field.flatten()  # Current temperature
        
        # Vectorized laser intensity calculation
        x_laser_min, x_laser_max, y_laser_min, y_laser_max = laser_indices
        
        # Create meshgrid of grid indices (i, j)
        i_indices = np.arange(self.N)
        j_indices = np.arange(self.N)
        I_grid, J_grid = np.meshgrid(i_indices, j_indices, indexing='ij')
        
        # Vectorized check: points in laser zone
        in_laser_zone = ((I_grid >= x_laser_min) & (I_grid <= x_laser_max) & 
                        (J_grid >= y_laser_min) & (J_grid <= y_laser_max))
        I_flat = np.where(in_laser_zone.flatten(), self.I0, 0.0)
        
        # Vectorized boundary condition calculation
        # Boundary points: i == 0 or i == N-1 or j == 0 or j == N-1
        is_boundary = ((I_grid == 0) | (I_grid == self.N-1) | 
                      (J_grid == 0) | (J_grid == self.N-1))
        bc_flat = np.where(is_boundary.flatten(), 1, 0)
        
        # Material properties are constant across grid
        k_flat = np.full(self.N * self.N, self.K)  # Thermal conductivity
        rho_flat = np.full(self.N * self.N, self.rho)  # Density
        C_flat = np.full(self.N * self.N, self.C)  # Specific heat capacity
        
        # Combine features: [x, y, k, rho, C, temperature, I, bc]
        features = np.column_stack([
            x_flat, y_flat, k_flat, rho_flat, C_flat, t_flat, I_flat, bc_flat
        ])
        
        return features

    def source_func(self, x, y):
        x_laser_min, x_laser_max, y_laser_min, y_laser_max = self.get_laser_indices()
        if (x >= x_laser_min and x <= x_laser_max and 
            y >= y_laser_min and y <= y_laser_max):
            return self.a * self.I0
        else:
            return 0.0

    def fdm_step(self, t_prev):
        """
        Perform one finite difference step for 2D heat equation (loop version).
        """
        Nx, Ny = t_prev.shape
        t_new = t_prev.copy()
        for x in range(1, Nx-1):
            for y in range(1, Ny-1):
                Dxx = (t_prev[x+1, y] - 2*t_prev[x, y] + t_prev[x-1, y]) / (self.dx**2)
                Dyy = (t_prev[x, y+1] - 2*t_prev[x, y] + t_prev[x, y-1]) / (self.dy**2)
                source = self.source_func(x, y)
                dTheta = (self.K/(self.rho*self.C)) * (Dxx + Dyy) + (1/(self.rho*self.C)) * source
                t_new[x, y] = t_prev[x, y] + self.dt * dTheta
        return t_new

    def fdm_step_matmul(self, t_prev):
        """
        Vectorized finite difference step using matrix multiplication.
        t_prev: (N, N) array
        L: (N*N, N*N) sparse Laplacian matrix
        """
        N = t_prev.shape[0]
        t_vec = t_prev.flatten()
        # Source term as a vector
        source_grid = np.fromfunction(np.vectorize(self.source_func), (N, N), dtype=int)
        source_vec = source_grid.flatten()
        # Forward Euler step
        t_new_vec = t_vec + self.dt * ((self.K/(self.rho*self.C)) * (self.L @ t_vec) / (self.dx**2) + (1/(self.rho*self.C)) * source_vec)
        t_new = t_new_vec.reshape((N, N))
        return t_new

    def plot_field(self, field, title, vmin=None, vmax=None):
        plt.figure(figsize=(8, 6))
        im = plt.contourf(self.X, self.Y, field, levels=50, cmap='hot', vmin=vmin, vmax=vmax)
        plt.colorbar(im, label='Temperature (K)')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title(title)
        xl_min, xl_max, yl_min, yl_max = self.laser.get_bounds()
        laser_zone = plt.Rectangle((xl_min, yl_min), xl_max-xl_min, yl_max-yl_min, 
                                  fill=False, edgecolor='white', linewidth=2, linestyle='--')
        plt.gca().add_patch(laser_zone)
        plt.text((xl_min + xl_max)/2, (yl_min + yl_max)/2, 'LASER ZONE', 
                 ha='center', va='center', color='white', fontweight='bold')
        plt.tight_layout()
        plt.show()

    @timing_decorator
    def solve(self, update_fn='loop'):
        t = np.full((self.N, self.N), self.theta0)
        self.apply_dirichlet_boundaries(t)
        time_steps = int(self.tf / self.dt)
        for i in range(time_steps):
            t_prev = t.copy()
            # Get laser indices at start of step (before it might move)
            laser_indices = self.get_laser_indices()
            
            if update_fn == 'loop':
                t = self.fdm_step(t_prev)
            elif update_fn == 'matmul':
                t = self.fdm_step_matmul(t_prev)
            else:
                raise ValueError('Unknown update_fn')
            self.apply_dirichlet_boundaries(t)
            
            # Update laser position every time step based on velocity
            self.laser.update_position(self.dt)
            if np.max(t) > 1e6:
                raise ValueError(f"Temperature too large at step {i}, unstable solution")
            
            if self.write_output:
                # Extract features from input state (t_prev) with laser position at start of step
                input_features = self.extract_features(t_prev, laser_indices)
                # Output is just the flattened next temperature
                output_temp = t.flatten().reshape(-1, 1)
                
                self.save_dict['input'].append(input_features)
                self.save_dict['output'].append(output_temp)
        return t

if __name__ == "__main__":
    # Realistic material properties for steel/laser processing
    I0 = 5e9      # Laser intensity (W/m²)
    a = 0.8       # Absorption coefficient
    K = 35        # Thermal conductivity (W/m·K)
    C = 500       # Specific heat capacity (J/kg·K)
    rho = 7700    # Density (kg/m³)
    theta_w = 300 # Wall temperature (K)
    theta0 = 300  # Initial temperature (K)
    dt = 0.1     # Time step (s)
    tf = 50      # Final time (s)
    N = 51        # Grid size
    l = 0.2      # Domain size (m)
    
    # Laser setup
    n_path = 128
    
    import random

    def pick_random(minmax):
        """Pick a random float in [min, max] given [min, max] list."""
        return random.uniform(minmax[0], minmax[1])

    # Define [min, max] for each parameter
    laser_position_x_range = [0.09, 0.11]
    laser_position_y_range = [0.09, 0.11]
    laser_dr_range = [0.002, 0.01]
    laser_velocity_range = [0.001, 0.02]  # m/s

    path_width_range = [0.08, 0.16]
    path_height_range = [0.06, 0.16]
    path_spacing_range = [0.02, 0.06]
    circle_radius_range = [0.04, 0.08]

    # Randomly pick values
    laser_position_x = pick_random(laser_position_x_range)
    laser_position_y = pick_random(laser_position_y_range)
    laser_dr = pick_random(laser_dr_range)
    laser_velocity = pick_random(laser_velocity_range)

    path_width = pick_random(path_width_range)
    path_height = pick_random(path_height_range)
    path_spacing = pick_random(path_spacing_range)
    circle_radius = pick_random(circle_radius_range)

    laser_position = (laser_position_x, laser_position_y)
    
    # Randomly pick a laser type from options
    laser_type_options = ['circle', 'rectangle', 'snake', 'single_point']
    laser_type = random.choice(laser_type_options)
    print(f"Randomly selected laser_type: {laser_type}")

    # Print selected laser path parameters
    print(f"Laser parameters:")
    print(f"  Position: {laser_position}")
    print(f"  dr (half-width): {laser_dr}")
    print(f"  velocity: {laser_velocity}")
    print(f"Path parameters:")
    print(f"  width: {path_width}")
    print(f"  height: {path_height}")
    print(f"  spacing: {path_spacing}")
    print(f"  circle_radius: {circle_radius}")
    print(f"  n_path: {n_path}")
    print(f"  laser_type: {laser_type}")

    if laser_type == 'snake':
        path_points = make_path(
            mode='snake', l=l, n_path=n_path, center=laser_position, width=path_width, height=path_height, spacing=path_spacing
        )
        laser = HeatSourceLaser(dr=laser_dr, path=path_points, velocity=laser_velocity)
    elif laser_type == 'rectangle':
        path_points = make_path(
            mode='rectangle', l=l, n_path=n_path, center=laser_position, width=path_width, height=path_height
        )
        laser = HeatSourceLaser(dr=laser_dr, path=path_points, velocity=laser_velocity)
    elif laser_type == 'circle':
        path_points = make_path(
            mode='circle', l=l, n_path=n_path, center=laser_position, radius=circle_radius
        )
        laser = HeatSourceLaser(dr=laser_dr, path=path_points, velocity=laser_velocity)
    elif laser_type == 'single_point':
        # Provide position for static case
        laser = HeatSourceLaser(dr=laser_dr, position=laser_position)
        path_points = None
    else:
        raise ValueError(f'Invalid laser type: {laser_type}')

    # Visualize the path
    print('Visualizing laser path...')
    laser.visualize_path(l, N)
    
    # Create save dictionary for output
    save_dict = {'input': [], 'output': []}
    
    # Create FDM_Laser instance with write_output enabled
    fdm = FDM_Laser(N, l, dt, tf, K, rho, C, theta0, theta_w, a, I0, laser, 
                    write_output=True, save_dict=save_dict)
    
    # Choose which method to use
    solve_method = 'matmul'  # or 'loop'
    print(f'Running {solve_method}-based FDM...')
    final_temp = fdm.solve(update_fn=solve_method)
    print(f"{solve_method.capitalize()}-based: max temp = {np.max(final_temp):.2f}, min temp = {np.min(final_temp):.2f}")
    
    
    # Plot final temperature field
    plt.figure(figsize=(8, 6))
    im = plt.contourf(fdm.X, fdm.Y, final_temp.T, levels=50, cmap='hot')
    plt.colorbar(im, label='Temperature (K)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f'{solve_method.capitalize()}-based FDM (t = {tf:.3f} s)')
    xl_min, xl_max, yl_min, yl_max = laser.get_bounds()
    laser_zone = plt.Rectangle((xl_min, yl_min), xl_max-xl_min, yl_max-yl_min, 
                              fill=False, edgecolor='white', linewidth=2, linestyle='--')
    plt.gca().add_patch(laser_zone)
    plt.text((xl_min + xl_max)/2, (yl_min + yl_max)/2, 'LASER ZONE', 
             ha='center', va='center', color='white', fontweight='bold')
    plt.tight_layout()
    plt.show() 

    # Verify input/output structure
    print("\n" + "="*60)
    print("DATASET STRUCTURE VERIFICATION")
    print("="*60)
    print(f"Number of time steps saved: {len(save_dict['input'])}")
    if len(save_dict['input']) > 0:
        print(f"Input shape: {save_dict['input'][0].shape}")
        print(f"Output shape: {save_dict['output'][0].shape}")
        print(f"Expected input shape: ({N*N}, 8) - [x, y, k, rho, C, temperature, I, bc]")
        print(f"Expected output shape: ({N*N}, 1) - [temperature_next]")
        
        # Check first input sample
        input_sample = save_dict['input'][0]
        print(f"\nFirst input sample statistics:")
        print(f"  x range: [{input_sample[:, 0].min():.4f}, {input_sample[:, 0].max():.4f}]")
        print(f"  y range: [{input_sample[:, 1].min():.4f}, {input_sample[:, 1].max():.4f}]")
        print(f"  k (constant): {input_sample[0, 2]:.2f}")
        print(f"  rho (constant): {input_sample[0, 3]:.2f}")
        print(f"  C (constant): {input_sample[0, 4]:.2f}")
        print(f"  temperature range: [{input_sample[:, 5].min():.2f}, {input_sample[:, 5].max():.2f}] K")
        print(f"  I (laser intensity): min={input_sample[:, 6].min():.2e}, max={input_sample[:, 6].max():.2e}, non-zero count={np.count_nonzero(input_sample[:, 6])}")
        print(f"  bc (boundary flag): 0 count={np.count_nonzero(input_sample[:, 7] == 0)}, 1 count={np.count_nonzero(input_sample[:, 7] == 1)}")
        
        # Check output sample
        output_sample = save_dict['output'][0]
        print(f"\nFirst output sample statistics:")
        print(f"  temperature_next range: [{output_sample.min():.2f}, {output_sample.max():.2f}] K")
    
    # Plot feature visualizations with interactive slider
    if len(save_dict['input']) > 0:
        num_steps = len(save_dict['input'])
        
        # Create figure with subplots and slider
        fig = plt.figure(figsize=(18, 13))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.1], hspace=0.3, wspace=0.3)
        axes = []
        for i in range(2):
            row = []
            for j in range(3):
                row.append(fig.add_subplot(gs[i, j]))
            axes.append(row)
        
        # Create slider axis
        slider_ax = fig.add_subplot(gs[2, :])
        
        # Store colorbars to remove them on update
        colorbars = [None] * 6
        
        # Initialize with first time step
        def update_plots(step_idx):
            step_idx = int(step_idx)
            input_sample = save_dict['input'][step_idx]
            output_sample = save_dict['output'][step_idx]
            
            # Reshape to 2D for visualization
            x_2d = input_sample[:, 0].reshape(N, N)
            y_2d = input_sample[:, 1].reshape(N, N)
            temp_2d = input_sample[:, 5].reshape(N, N)
            I_2d = input_sample[:, 6].reshape(N, N)
            bc_2d = input_sample[:, 7].reshape(N, N)
            temp_next_2d = output_sample[:, 0].reshape(N, N)
            temp_diff = temp_next_2d - temp_2d
            k_2d = input_sample[:, 2].reshape(N, N)
            
            # Remove old colorbars
            for cb in colorbars:
                if cb is not None:
                    cb.remove()
            
            # Clear and update each subplot
            for ax in axes[0] + axes[1]:
                ax.clear()
            
            # Temperature (input)
            im1 = axes[0][0].contourf(x_2d, y_2d, temp_2d, levels=50, cmap='hot')
            axes[0][0].set_title(f'Input: Temperature (K) - Step {step_idx}')
            axes[0][0].set_xlabel('x (m)')
            axes[0][0].set_ylabel('y (m)')
            colorbars[0] = plt.colorbar(im1, ax=axes[0][0])
            
            # Laser intensity
            im2 = axes[0][1].contourf(x_2d, y_2d, I_2d, levels=20, cmap='Reds')
            axes[0][1].set_title(f'Input: Laser Intensity (W/m²)\nNon-zero points: {np.count_nonzero(I_2d)}')
            axes[0][1].set_xlabel('x (m)')
            axes[0][1].set_ylabel('y (m)')
            colorbars[1] = plt.colorbar(im2, ax=axes[0][1])
            
            # Boundary condition flag
            im3 = axes[0][2].imshow(bc_2d, extent=[0, l, 0, l], origin='lower', cmap='RdYlGn', vmin=0, vmax=1)
            axes[0][2].set_title('Input: Boundary Condition Flag\n(0=interior, 1=Dirichlet)')
            axes[0][2].set_xlabel('x (m)')
            axes[0][2].set_ylabel('y (m)')
            colorbars[2] = plt.colorbar(im3, ax=axes[0][2])
            
            # Temperature (output)
            im4 = axes[1][0].contourf(x_2d, y_2d, temp_next_2d, levels=50, cmap='hot')
            axes[1][0].set_title('Output: Temperature Next (K)')
            axes[1][0].set_xlabel('x (m)')
            axes[1][0].set_ylabel('y (m)')
            colorbars[3] = plt.colorbar(im4, ax=axes[1][0])
            
            # Temperature difference
            im5 = axes[1][1].contourf(x_2d, y_2d, temp_diff, levels=50, cmap='coolwarm')
            axes[1][1].set_title('Temperature Change (Next - Current)')
            axes[1][1].set_xlabel('x (m)')
            axes[1][1].set_ylabel('y (m)')
            colorbars[4] = plt.colorbar(im5, ax=axes[1][1])
            
            # Material properties (showing k as example, all are constant)
            im6 = axes[1][2].imshow(k_2d, extent=[0, l, 0, l], origin='lower', cmap='viridis')
            axes[1][2].set_title(f'Input: Thermal Conductivity k (W/m·K)\nConstant: {k_2d[0, 0]:.2f}')
            axes[1][2].set_xlabel('x (m)')
            axes[1][2].set_ylabel('y (m)')
            colorbars[5] = plt.colorbar(im6, ax=axes[1][2])
            
            fig.canvas.draw_idle()
        
        # Create slider
        slider = Slider(slider_ax, 'Time Step', 0, num_steps - 1, valinit=0, valfmt='%d')
        slider.on_changed(update_plots)
        
        # Initialize with first step
        update_plots(0)
        
        plt.show()
        
        # Plot feature distributions with interactive slider
        fig2 = plt.figure(figsize=(12, 11))
        gs2 = fig2.add_gridspec(3, 2, height_ratios=[1, 1, 0.1], hspace=0.3, wspace=0.3)
        axes2 = []
        for i in range(2):
            row = []
            for j in range(2):
                row.append(fig2.add_subplot(gs2[i, j]))
            axes2.append(row)
        
        slider_ax2 = fig2.add_subplot(gs2[2, :])
        
        def update_distributions(step_idx):
            step_idx = int(step_idx)
            input_sample = save_dict['input'][step_idx]
            output_sample = save_dict['output'][step_idx]
            
            # Reshape to 2D for temp_diff calculation
            temp_2d = input_sample[:, 5].reshape(N, N)
            temp_next_2d = output_sample[:, 0].reshape(N, N)
            temp_diff = temp_next_2d - temp_2d
            
            # Clear and update each subplot
            for ax in axes2[0] + axes2[1]:
                ax.clear()
            
            # Temperature distribution
            axes2[0][0].hist(input_sample[:, 5], bins=50, alpha=0.7, label='Input temp', color='blue')
            axes2[0][0].hist(output_sample[:, 0], bins=50, alpha=0.7, label='Output temp', color='orange')
            axes2[0][0].set_xlabel('Temperature (K)')
            axes2[0][0].set_ylabel('Frequency')
            axes2[0][0].set_title(f'Temperature Distribution - Step {step_idx}')
            axes2[0][0].legend()
            axes2[0][0].grid(True, alpha=0.3)
            
            # Laser intensity distribution
            I_all = input_sample[:, 6]
            I_nonzero = input_sample[input_sample[:, 6] > 0, 6]
            axes2[0][1].hist(I_all, bins=20, alpha=0.7, color='red')
            axes2[0][1].set_xlabel('Laser Intensity (W/m²)')
            axes2[0][1].set_ylabel('Frequency')
            axes2[0][1].set_title(f'Laser Intensity Distribution\n({len(I_all)} total points, {len(I_nonzero)} in laser zone)')
            axes2[0][1].grid(True, alpha=0.3)
            
            # Boundary condition distribution
            bc_counts = np.bincount(input_sample[:, 7].astype(int))
            axes2[1][0].bar(['Interior (0)', 'Boundary (1)'], [bc_counts[0], bc_counts[1]], 
                            color=['green', 'red'], alpha=0.7)
            axes2[1][0].set_ylabel('Count')
            axes2[1][0].set_title('Boundary Condition Distribution')
            axes2[1][0].grid(True, alpha=0.3, axis='y')
            
            # Temperature change distribution
            axes2[1][1].hist(temp_diff.flatten(), bins=50, alpha=0.7, color='purple')
            axes2[1][1].set_xlabel('Temperature Change (K)')
            axes2[1][1].set_ylabel('Frequency')
            axes2[1][1].set_title('Temperature Change Distribution')
            axes2[1][1].axvline(0, color='black', linestyle='--', linewidth=2)
            axes2[1][1].grid(True, alpha=0.3)
            
            fig2.canvas.draw_idle()
        
        # Create slider for distributions
        slider2 = Slider(slider_ax2, 'Time Step', 0, num_steps - 1, valinit=0, valfmt='%d')
        slider2.on_changed(update_distributions)
        
        # Initialize with first step
        update_distributions(0)
        
        plt.show()