import numpy as np
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed
lam = 1.3e-1
v = 2.098e-17
xi = 1e34
Mp = 1.22e19

def ode_sistema_original(t, Y):
    y, vel = Y
    esc = 10e6
    #potencial = 0.5*(1 + np.exp(-2*y))**(-2)
    potencial = (Mp**4*lam/4)*((1/xi)*np.exp(2*y) - v**2)**2/(1 + np.exp(2*y))**2
    damping = (3*xi*vel/esc)*np.sqrt( (1/12)* ( (np.exp(2*y)/xi - v**2)**2 / (1 + np.exp(2*y))**2)  + (vel**2)/(esc**2*xi**2))
    forcing = (1/6)*( np.exp(2*y)*(np.exp(2*y) - xi*v**2)*(1+xi*v**2) )/(1 + np.exp(2*y))**3
    dv_dt =esc**2*(-damping - forcing) ## Esta es la ecuación diferencial completa.
    return [vel, dv_dt]


def ode_sistema(t, Y):
    y, vel = Y 
    exp2y = np.exp(2*y)
    denom = (1 + exp2y)
    
    damping = 3*xi*vel*np.sqrt((1/12)*((exp2y/xi - v**2)**2 / denom**2) + (vel**2)/xi**2
    )
    forcing = (1/6) * (exp2y*(exp2y - xi*v**2)*(1+xi*v**2)) / denom**3
    
    dv_dt = -damping - forcing
    return [vel, dv_dt]

    
def solve_single_ode(t_span, t_eval, y0, v0):
    sol = solve_ivp(ode_sistema, t_span, [y0, v0], t_eval=t_eval)
    return sol.y[0], sol.y[1] 

def solve_ODEs_parallel(t_span, t_eval, y0_array, v0_array, n_jobs=-1, batch_size=100, save=False, save_dir="../data/ode_results"):
    """
    Solve ODEs in parallel and save results as .npy files.
    
    Args:
        save (bool): If True, saves arrays as individual .npy files
        save_dir (str): Directory to save files (will be created if nonexistent)
    """
    # Create grid of initial conditions
    y0_comb, v0_comb = np.meshgrid(y0_array, v0_array, indexing='ij')
    Y0_comb = np.column_stack([y0_comb.ravel(), v0_comb.ravel()])
    n_combinations = len(Y0_comb)

    # Process batches
    results = []
    for i in range(0, n_combinations, batch_size):
        batch = Y0_comb[i:i + batch_size]
        batch_results = Parallel(n_jobs=n_jobs)(
            delayed(solve_single_ode)(t_span, t_eval, y0, v0)
            for y0, v0 in batch
        )
        results.extend(batch_results)
        print(f"\rProcessed {min(i + batch_size, n_combinations)}/{n_combinations}", end="", flush=True)

    # Convert to arrays
    y = np.array([r[0] for r in results]).T  # shape: (n_times, n_combinations)
    v = np.array([r[1] for r in results]).T

    # Compute derived quantities
    potencial = 0.5 * (1 + np.exp(-2 * y)) ** (-2)
    damping = 3 * v * np.sqrt(potencial + v ** 2)
    forcing = np.exp(-2 * y) / (1 + np.exp(-2 * y)) ** 3
    acc = -damping - forcing

    # Package results
    result_dict = {
        'y': y,
        'vel': v,
        'acc': acc,
        'potencial': potencial,
        'damping': damping,
        'forcing': forcing,
    }

    if save:
        import os 
        os.makedirs(save_dir, exist_ok=True)
        
        np.save(os.path.join(save_dir, 'field.npy'), y)
        np.save(os.path.join(save_dir, 'field_derivative.npy'), v)
        np.save(os.path.join(save_dir, 'field_second_derivative.npy'), acc)
        np.save(os.path.join(save_dir, 'potencial.npy'), potencial)
        np.save(os.path.join(save_dir, 'damping.npy'), damping)
        np.save(os.path.join(save_dir, 'forcing.npy'), forcing)
        
        print(f"\nResults saved to {save_dir}/")

    return result_dict

if __name__ == "__main__":
    print("This module contains the calculate_epsilon function for inflation calculations.")
    print("Import it in your notebook using:")
    print("from epsilon_calculator import calculate_epsilon")