# Adaptive Quantum Gate Calibration Framework

## Overview
This repository contains a sample-efficient framework for adaptive quantum gate calibration using Bayesian optimization and optimal control techniques. The framework is designed to significantly reduce the measurement overhead required for high-fidelity quantum gate calibration, making it particularly valuable for fault-tolerant quantum computing applications.

## Key Features
- **Bayesian Optimization**: Uses Gaussian Process regression to model the relationship between control parameters and gate fidelity
- **Multiple Acquisition Functions**: Implements UCB, Thompson sampling, and Expected Improvement strategies
- **Intelligent Sampling**: Utilizes Latin Hypercube Sampling for efficient parameter space exploration
- **Comparative Analysis**: Benchmarks Bayesian optimization against random and grid search methods
- **Advanced Visualization**: Includes comprehensive tools for visualizing optimization progress and fidelity surfaces
- **Quantum Simulation**: Leverages QuTiP for time evolution simulation of quantum gate operations

## Requirements
```
numpy
matplotlib
scipy
scikit-learn
qutip
tqdm
```


## Usage
Basic usage example:
```python
from quantumgate import AdaptiveGateCalibration, demo_gate_calibration

# Run a demonstration with default settings
calibrator, results = demo_gate_calibration()

# Or create a custom calibration task
import numpy as np
import qutip as qt

# Define a custom target gate (e.g., Hadamard gate)
target_gate = (qt.sigmax() + qt.sigmaz()).unit().full()

# Initialize the calibrator with custom settings
calibrator = AdaptiveGateCalibration(
    dim=2,
    target_gate=target_gate,
    noise_level=0.02,
    seed=42
)

# Run optimization
best_params, best_fidelity = calibrator.run_optimization(
    n_iterations=30,
    initial_measurements=10,
    acq_method='ucb'
)

# Visualize results
calibrator.plot_optimization_results()
```

## Method Comparison
The framework allows you to compare different optimization approaches:
```python
results = calibrator.compare_optimization_methods(
    methods=['random', 'grid', 'bayesian'], 
    n_iterations=25,
    n_runs=3
)
```

## How It Works
1. The framework models a quantum system with a drift Hamiltonian and two control Hamiltonians
2. Control pulse parameters (amplitudes and durations) are optimized to produce a unitary operation matching the target gate
3. Bayesian optimization adaptively selects the most promising parameters to test based on previous measurements
4. Simulated experimental noise is included to mimic real-world conditions
5. Optimization proceeds iteratively, gradually improving gate fidelity while minimizing the required measurements

## Applications
- Calibration of quantum gates in superconducting quantum processors
- Optimization of control pulses for trapped-ion quantum computers
- Development of error-robust gate operations
- Research into sample-efficient characterization of quantum devices

## Citation
If you use this code in your research, please cite:
```
@misc{QuantumGateCalibration2025,
  author = {Your Name},
  title = {Sample-Efficient Adaptive Quantum Gate Calibration Framework},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/username/quantum-gate-calibration}}
}
```

## License
MIT

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
