# A-Efolds: Numerical Calculation of e-folds in Higgs Inflation

This repository contains Python scripts and Jupyter Notebooks for the numerical calculation of e-folds in the context of Higgs Inflation cosmology.

## Repository Structure

The project is organized as follows:

- **`src/`**: Contains the core Python modules for solving Ordinary Differential Equations (ODEs).
  - `ODE_solver.py`: Main solver functions.
  - `ODE_solver_scaled_tau.py`: Solver with scaled time variable.
- **`notebooks/`**: Jupyter notebooks for analysis and visualization.
  - `numerical_efolds(Higgs inflation).ipynb`: Main analysis notebook.
  - `InitialConditionSpace.ipynb`: Exploration of initial conditions.
  - `(tau escalado)numerical_efolds(Higgs inflation).ipynb`: Analysis using scaled time.
- **`data/`**: Directory for storing output data files (e.g., `.npz`, `.npy`).
  - *Note: Data files are excluded from version control.*
- **`images/`**: Directory for storing generated figures.
  - *Note: Generated images are excluded from version control.*
- **`logs/`**: Directory for execution logs.

## Installation

1. Clone the repository.
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Navigate to the `notebooks/` directory.
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open and run the desired notebook. The notebooks are configured to automatically import modules from the `src/` directory.

## Output

- **Data**: Numerical results are saved to `data/ode_results/` or as `.npz` files in `data/`.
- **Figures**: Plots and figures are saved to `images/` or `images/figuras/`.

## License

[Insert License Here]
