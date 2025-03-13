import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
import qutip as qt
from tqdm import tqdm

class AdaptiveGateCalibration:
    """
    Enhanced framework for sample-efficient adaptive quantum gate calibration using
    Bayesian optimization and optimal control techniques.
    """
    def __init__(self, dim=2, target_gate=None, noise_level=0.01, seed=42):
        """
        Initialize the adaptive gate calibration framework.
        
        Parameters:
        -----------
        dim : int
            Dimension of the quantum system (default: 2 for a qubit)
        target_gate : np.ndarray
            Target gate unitary matrix
        noise_level : float
            Simulated experimental noise level
        seed : int
            Random seed for reproducibility
        """
        np.random.seed(seed)
        self.dim = dim
        
        # Default target gate: X gate (NOT gate)
        if target_gate is None:
            self.target_gate = qt.sigmax().full()
        else:
            self.target_gate = target_gate
            
        self.noise_level = noise_level
        
        # Enhanced parameter bounds for better exploration
        self.param_bounds = np.array([
            [-2.0, 2.0],   # amplitude_x: increased range
            [-2.0, 2.0],   # amplitude_y: increased range
            [0.05, 5.0],   # duration_x: wider range
            [0.05, 5.0]    # duration_y: wider range
        ])
        self.param_names = ['amplitude_x', 'amplitude_y', 'duration_x', 'duration_y']
        
        # System Hamiltonian components
        self.H0 = qt.sigmaz() * 2 * np.pi  # Drift Hamiltonian
        self.Hx = qt.sigmax() * 2 * np.pi  # Control Hamiltonian in x
        self.Hy = qt.sigmay() * 2 * np.pi  # Control Hamiltonian in y
        
        # Gate fidelity evaluation history
        self.param_history = []
        self.fidelity_history = []
        
        # Bayesian optimization setup
        self.gp = None
        self.init_gp()
        
    def init_gp(self):
        """Initialize the Gaussian Process Regressor with improved hyperparameters."""
        # Modified kernel parameters for better exploration
        kernel = ConstantKernel(1.0, constant_value_bounds=(1e-1, 1e3)) * \
                Matern(length_scale=[1.0] * 4, nu=2.5, length_scale_bounds=(0.01, 100.0))
        
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.noise_level**2,
            normalize_y=True,
            n_restarts_optimizer=25  # Increased from 10
        )
    
    def pulse_to_unitary(self, params):
        """
        Convert control pulse parameters to a unitary gate.
        
        Parameters:
        -----------
        params : np.ndarray
            Control pulse parameters [amplitude_x, amplitude_y, duration_x, duration_y]
            
        Returns:
        --------
        U : np.ndarray
            Resulting unitary matrix
        """
        amp_x, amp_y, duration_x, duration_y = params
        
        def h_t(t, args):
            H = self.H0.copy()
            
            # X pulse
            if t < duration_x:
                H += amp_x * self.Hx
                
            # Y pulse
            if duration_x <= t < (duration_x + duration_y):
                H += amp_y * self.Hy
                
            return H
        
        # Solve time evolution with increased time points for better accuracy
        tlist = np.linspace(0, duration_x + duration_y, 200)  # Increased from 100
        result = qt.sesolve(h_t, qt.qeye(self.dim), tlist)
        
        # Extract final unitary
        U = result.states[-1].full()
        return U
    
    def compute_gate_fidelity(self, params, n_measurements=100):
        """
        Compute the gate fidelity with respect to the target gate.
        Simulates experimental noise and measurement overhead.
        
        Parameters:
        -----------
        params : np.ndarray
            Control pulse parameters
        n_measurements : int
            Number of simulated measurements for fidelity estimation
            
        Returns:
        --------
        fidelity : float
            Estimated gate fidelity
        """
        U = self.pulse_to_unitary(params)
        
        # Calculate true process fidelity
        Utarget_dag = self.target_gate.conj().T
        true_fidelity = np.abs(np.trace(Utarget_dag @ U) / self.dim)**2
        
        # Simulate experimental noise based on number of measurements
        noise = np.random.normal(0, self.noise_level / np.sqrt(n_measurements))
        measured_fidelity = max(0, min(1, true_fidelity + noise))
        
        return measured_fidelity
    
    def acquisition_function(self, params, method='ucb', kappa=2.0):
        """
        Enhanced acquisition function with multiple strategies.
        
        Parameters:
        -----------
        params : np.ndarray
            Control pulse parameters to evaluate
        method : str
            Acquisition function type ('ucb', 'ei', 'pi', 'thompson')
        kappa : float
            Exploration-exploitation trade-off parameter
            
        Returns:
        --------
        acq_value : float
            Acquisition function value
        """
        params = params.reshape(1, -1)
        mu, sigma = self.gp.predict(params, return_std=True)
        
        if method == 'ucb':
            # Dynamic exploration-exploitation trade-off
            n_samples = len(self.fidelity_history)
            kappa_dynamic = kappa * np.sqrt(np.log(n_samples + 1)) if n_samples > 0 else kappa
            return mu + kappa_dynamic * sigma
        elif method == 'thompson':
            # Thompson Sampling
            return np.random.normal(mu, sigma)
        elif method == 'ei':  # Expected Improvement
            best_f = np.max(self.fidelity_history) if self.fidelity_history else 0
            imp = mu - best_f
            Z = imp / (sigma + 1e-6)
            return imp * (0.5 * (1 + np.math.erf(Z / np.sqrt(2))))
        elif method == 'pi':  # Probability of Improvement
            best_f = np.max(self.fidelity_history) if self.fidelity_history else 0
            Z = (mu - best_f) / (sigma + 1e-6)
            return 0.5 * (1 + np.math.erf(Z / np.sqrt(2)))
        else:
            raise ValueError(f"Unknown acquisition function: {method}")
    
    def next_point_to_sample(self, method='ucb', n_restarts=15):
        """
        Improved optimization strategy with multi-starting points.
        
        Parameters:
        -----------
        method : str
            Acquisition function type
        n_restarts : int
            Number of random restarts for optimization
            
        Returns:
        --------
        best_params : np.ndarray
            Next parameters to sample
        """
        best_params = None
        best_acq = -np.inf
        
        # Generate better starting points using Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=len(self.param_bounds))
        start_points = sampler.random(n=n_restarts)
        
        # Scale to parameter bounds
        for i in range(len(self.param_bounds)):
            start_points[:, i] = start_points[:, i] * (self.param_bounds[i, 1] - self.param_bounds[i, 0]) + self.param_bounds[i, 0]
        
        for x0 in start_points:
            # Use different optimization methods
            for opt_method in ['L-BFGS-B', 'SLSQP']:
                res = minimize(
                    lambda x: -self.acquisition_function(x, method=method),
                    x0,
                    bounds=self.param_bounds,
                    method=opt_method
                )
                
                if -res.fun > best_acq:
                    best_acq = -res.fun
                    best_params = res.x
                    
        return best_params
    
    def update_model(self, params, fidelity):
        """
        Update the Gaussian Process model with new data.
        
        Parameters:
        -----------
        params : np.ndarray
            Control pulse parameters
        fidelity : float
            Measured fidelity value
        """
        self.param_history.append(params)
        self.fidelity_history.append(fidelity)
        
        X = np.array(self.param_history)
        y = np.array(self.fidelity_history)
        
        # Update the GP model
        self.gp.fit(X, y)
    
    def run_optimization(self, n_iterations=30, initial_measurements=10, acq_method='ucb'):
        """
        Enhanced optimization with better initialization and adaptive sampling.
        
        Parameters:
        -----------
        n_iterations : int
            Number of optimization iterations
        initial_measurements : int
            Number of initial random measurements
        acq_method : str
            Acquisition function type
            
        Returns:
        --------
        best_params : np.ndarray
            Optimal control parameters
        best_fidelity : float
            Best achieved fidelity
        """
        # Initial random sampling using Latin Hypercube
        sampler = qmc.LatinHypercube(d=len(self.param_bounds))
        initial_points = sampler.random(n=initial_measurements)
        
        for i in range(initial_measurements):
            params = np.zeros(len(self.param_bounds))
            for j in range(len(self.param_bounds)):
                params[j] = initial_points[i, j] * (self.param_bounds[j, 1] - self.param_bounds[j, 0]) + self.param_bounds[j, 0]
            
            fidelity = self.compute_gate_fidelity(params, n_measurements=200)  # More measurements initially
            self.update_model(params, fidelity)
            
        # Adaptive optimization loop
        pbar = tqdm(range(n_iterations), desc="Optimizing gate parameters")
        for i in pbar:
            # Alternate between UCB and Thompson Sampling
            current_method = 'ucb' if i % 2 == 0 else 'thompson'
            
            next_params = self.next_point_to_sample(method=current_method)
            # Increase measurements as optimization progresses
            n_measurements = 100 + int(i * 10)  # Gradually increase precision
            next_fidelity = self.compute_gate_fidelity(next_params, n_measurements=n_measurements)
            self.update_model(next_params, next_fidelity)
            
            best_idx = np.argmax(self.fidelity_history)
            best_fid = self.fidelity_history[best_idx]
            pbar.set_postfix({"best_fidelity": f"{best_fid:.6f}"})
        
        # Return best parameters and fidelity
        best_idx = np.argmax(self.fidelity_history)
        return self.param_history[best_idx], self.fidelity_history[best_idx]
    
    # ... (previous code remains the same until the incomplete plot_optimization_results method)

    def plot_optimization_results(self):
        """Plot the optimization results including fidelity improvement over iterations."""
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Fidelity vs Iteration
        plt.subplot(2, 2, 1)
        plt.plot(range(len(self.fidelity_history)), self.fidelity_history, 'o-', color='blue')
        plt.axhline(y=1.0, linestyle='--', color='red')
        plt.title('Gate Fidelity vs Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Fidelity')
        plt.grid(True)
        
        # Plot 2: Parameter trajectories
        plt.subplot(2, 2, 2)
        params_array = np.array(self.param_history)
        for i, name in enumerate(self.param_names):
            plt.plot(range(len(params_array)), params_array[:, i], 'o-', label=name)
        plt.title('Parameter Values vs Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Parameter Value')
        plt.legend()
        plt.grid(True)
        
        # If we have enough data, create prediction surface plots
        if len(self.param_history) >= 10:
            best_idx = np.argmax(self.fidelity_history)
            best_params = self.param_history[best_idx]
            
            # Plot 3: Predicted fidelity surface (amplitude_x vs amplitude_y)
            plt.subplot(2, 2, 3)
            xx, yy = np.meshgrid(
                np.linspace(self.param_bounds[0, 0], self.param_bounds[0, 1], 20),
                np.linspace(self.param_bounds[1, 0], self.param_bounds[1, 1], 20)
            )
            
            fixed_params = np.tile(best_params, (xx.size, 1))
            fixed_params[:, 0] = xx.ravel()
            fixed_params[:, 1] = yy.ravel()
            
            predicted_fidelity = self.gp.predict(fixed_params)
            predicted_fidelity = predicted_fidelity.reshape(xx.shape)
            
            plt.contourf(xx, yy, predicted_fidelity, levels=50, cmap='viridis')
            plt.colorbar(label='Predicted Fidelity')
            plt.plot(best_params[0], best_params[1], 'ro', markersize=10, label='Optimum')
            plt.title('Predicted Fidelity (amplitude_x vs amplitude_y)')
            plt.xlabel('amplitude_x')
            plt.ylabel('amplitude_y')
            plt.legend()
            
            # Plot 4: Predicted fidelity surface (duration_x vs duration_y)
            plt.subplot(2, 2, 4)
            xx, yy = np.meshgrid(
                np.linspace(self.param_bounds[2, 0], self.param_bounds[2, 1], 20),
                np.linspace(self.param_bounds[3, 0], self.param_bounds[3, 1], 20)
            )
            
            fixed_params = np.tile(best_params, (xx.size, 1))
            fixed_params[:, 2] = xx.ravel()
            fixed_params[:, 3] = yy.ravel()
            
            predicted_fidelity = self.gp.predict(fixed_params)
            predicted_fidelity = predicted_fidelity.reshape(xx.shape)
            
            plt.contourf(xx, yy, predicted_fidelity, levels=50, cmap='viridis')
            plt.colorbar(label='Predicted Fidelity')
            plt.plot(best_params[2], best_params[3], 'ro', markersize=10, label='Optimum')
            plt.title('Predicted Fidelity (duration_x vs duration_y)')
            plt.xlabel('duration_x')
            plt.ylabel('duration_y')
            plt.legend()
        
        plt.tight_layout()
        plt.show()

    def compare_optimization_methods(self, methods=['random', 'grid', 'bayesian'], 
                                  n_iterations=25, n_runs=3):
        """
        Compare the efficiency of different optimization methods.
        
        Parameters:
        -----------
        methods : list
            List of methods to compare
        n_iterations : int
            Number of iterations for each method
        n_runs : int
            Number of runs for statistical comparison
            
        Returns:
        --------
        results : dict
            Dictionary of results for each method
        """
        results = {method: {'fidelities': [], 'params': []} for method in methods}
        
        for method in methods:
            print(f"\nTesting method: {method}")
            
            for run in range(n_runs):
                print(f"  Run {run+1}/{n_runs}")
                
                if method == 'random':
                    # Random search with Latin Hypercube Sampling
                    sampler = qmc.LatinHypercube(d=len(self.param_bounds))
                    points = sampler.random(n=n_iterations)
                    best_fidelity = 0
                    best_params = None
                    
                    for i in tqdm(range(n_iterations), desc="Random search"):
                        params = np.zeros(len(self.param_bounds))
                        for j in range(len(self.param_bounds)):
                            params[j] = points[i, j] * (self.param_bounds[j, 1] - self.param_bounds[j, 0]) + self.param_bounds[j, 0]
                        
                        fidelity = self.compute_gate_fidelity(params)
                        if fidelity > best_fidelity:
                            best_fidelity = fidelity
                            best_params = params
                    
                elif method == 'grid':
                    # Grid search with adaptive resolution
                    points_per_dim = max(2, int(n_iterations**(1/4)))
                    best_fidelity = 0
                    best_params = None
                    
                    # Create grid points using Latin Hypercube Sampling
                    sampler = qmc.LatinHypercube(d=len(self.param_bounds))
                    points = sampler.random(n=min(n_iterations, points_per_dim**4))
                    
                    for point in tqdm(points, desc="Grid search"):
                        params = np.zeros(len(self.param_bounds))
                        for j in range(len(self.param_bounds)):
                            params[j] = point[j] * (self.param_bounds[j, 1] - self.param_bounds[j, 0]) + self.param_bounds[j, 0]
                        
                        fidelity = self.compute_gate_fidelity(params)
                        if fidelity > best_fidelity:
                            best_fidelity = fidelity
                            best_params = params
                
                elif method == 'bayesian':
                    # Reset for this run
                    self.param_history = []
                    self.fidelity_history = []
                    self.init_gp()
                    
                    # Run Bayesian optimization
                    best_params, best_fidelity = self.run_optimization(
                        n_iterations=n_iterations-5,
                        initial_measurements=5,
                        acq_method='ucb'
                    )
                
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                # Store results
                results[method]['fidelities'].append(best_fidelity)
                results[method]['params'].append(best_params)
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        
        for method in methods:
            fids = results[method]['fidelities']
            plt.bar(method, np.mean(fids), yerr=np.std(fids), capsize=5, 
                   alpha=0.7, label=method)
        
        plt.ylabel('Best Fidelity Achieved')
        plt.title('Comparison of Optimization Methods')
        plt.ylim([0, 1.0])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return results


def demo_gate_calibration():
    """Run a demonstration of the enhanced adaptive gate calibration framework."""
    print("Initializing Enhanced Adaptive Gate Calibration Framework...")
    
    # Create X gate calibration task
    calibrator = AdaptiveGateCalibration(dim=2)
    
    print("\nRunning Bayesian optimization for gate calibration...")
    best_params, best_fidelity = calibrator.run_optimization(
        n_iterations=30,
        initial_measurements=10
    )
    
    print(f"\nBest parameters found: {best_params}")
    print(f"Best fidelity achieved: {best_fidelity:.6f}")
    
    # Plot results
    print("\nPlotting optimization results...")
    calibrator.plot_optimization_results()
    
    # Compare with other methods
    print("\nComparing optimization methods...")
    results = calibrator.compare_optimization_methods(
        methods=['random', 'grid', 'bayesian'], 
        n_iterations=25,
        n_runs=3
    )
    
    # Show best parameters for each method
    print("\nSummary of optimization methods:")
    for method, data in results.items():
        avg_fidelity = np.mean(data['fidelities'])
        std_fidelity = np.std(data['fidelities'])
        print(f"{method.capitalize()}: {avg_fidelity:.6f} ± {std_fidelity:.6f}")
    
    return calibrator, results


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run demonstration with enhanced parameters
    print(f"Starting quantum gate calibration demo")
    
    calibrator, results = demo_gate_calibration()