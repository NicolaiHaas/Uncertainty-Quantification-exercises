import time

import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.helpers import compute_errors, generate_grid, load_reference, simulate
from utils.interpolation import FirstBarycentricLagrange

dt = .1 # model parameter as defined in exercise 1

'''
    Samples from the omega/frequency distribution and calculates the solution for each frequency at time
    target_t.
    Args:
        omega_distr: cp.Distribution        - The chaospy distribution over the omegas
        n_sampels: int                      - The number of samples to draw.
        target_t: float                     - The time at which model solutions are calculated.
        model_kwargs: dict[str -> float]    - The model arguments for the oscillator ODE. See oscillator.py
        init_cond: dict[str -> float]       - The model's intial conditions.
    
    Returns:
        solutions: np.array                 - The oscillators amplitude at time target_t for all sampled
                                              frequencies
'''
def estimate_monte_carlo(
    omega_distr: cp.Distribution,
    n_samples: int,
    target_t: float,
    model_kwargs: dict[str, float],
    init_cond: dict[str, float],
) -> npt.NDArray:
    # ====================================================================
    omega_samples = omega_distr.sample(n_samples)
    time_grid = np.arange(0,target_t+dt,dt)
    solutions = simulate(np.array(time_grid),omega_samples,model_kwargs,init_cond)[:,-1]
    # ====================================================================
    return solutions

'''
    Takes upper and lower values of Omega / Frequencies and constructs for a given point in
    time a Lagrange interpolation of the omega -> oscillator_amplitude mapping.
    Args:
        omega_bounds: (float,float)         - The upper and lower bound of interpolation nodes for omega.
        n_nodes: int                        - Number of interpolation nodes.
        target_t: float                     - The point in time for which model solutions are calculated /
                                              the fixed t for which the omega interpolation holds
        model_kwargs: dict[str -> float]    - The model arguments for the oscillator ODE. See oscillator.py
        init_cond: dict[str -> float]       - The model's intial conditions.
    
    Returns:
        interpolator: FirstBarycentricLagrange - The Lagrange interpolator object for the 
                                                 f_t(omega) = amplitude function.
'''
def fit_lagrange(
    omega_bounds: tuple[float, float],
    n_nodes: int,
    target_t: float,
    model_kwargs: dict[str, float],
    init_cond: dict[str, float],
) -> FirstBarycentricLagrange:
    # ====================================================================
    # Generate nodes for interpolating f(omega)
    nodes = generate_grid(omega_bounds,n_nodes,grid_type='chebyshev')

    # create ODE solutions at target time for every interpolation node
    time_grid = np.arange(0,target_t+dt,dt)
    model_solutions = simulate(time_grid,nodes,model_kwargs,init_cond)[:,-1]
    
    # Construct the interpolator
    interpolator = FirstBarycentricLagrange(nodes=nodes,values=model_solutions)
    # ====================================================================
    return interpolator
    

'''
    Interpolate the function f_t(omega) = amplitude at randomly sampled omega values. Parameter t is
    defined in the interpolator object.
    Args:
        interpolator: FirstBarycentricLagrange  - The inerpolator for the aforementioned function.
        omega_distr: cp.Distribution            - The chaospy distribution assumed on Omega
        n_samples: int                          - The number of samples of omega
        
    Returns:
        solutions: np.array                     - The solutions f_t(omega) for the sampled values.
'''
def evaluate_pce(
    interpolator: FirstBarycentricLagrange, omega_distr: cp.Distribution, n_samples: int
) -> npt.NDArray:
    # ====================================================================
    x_omega = omega_distr.sample(n_samples)
    solutions = interpolator.evaluate(x_omega)
    # ====================================================================
    return solutions

if __name__ == "__main__":
    # ====================================================================
    np.random.seed(42) # also counts for chaospy

    # Model parameters
    t_bounds = (0,10)
    c = 0.5
    k = 2
    f = 0.5
    y0 = 0.5
    y1 = 0
    omega_bounds = (0.95,1.05)
    omega_dist = cp.Uniform(omega_bounds[0],omega_bounds[1])
    target_t = t_bounds[1]
    model_kwargs = {
        "c":c,
        "k":k,
        "f":f    
    }
    init_cond = {
        "y0":y0,
        "y1":y1
    }

    # Sampling parameters
    N_ls = [2,5,10,20]
    M_ls = [10,100,1e3,1e4]

    # Reference values
    ref_mean, ref_var = load_reference("oscillator_ref.txt")
    # ====================================================================
    # Lagrange Interpolation calculation
    # ====================================================================
    sample_ls_PCE = []
    time_ls_PCE = []
    
    start_time_PCE = time.time()
    for i,n in enumerate(N_ls):
        interpolator = fit_lagrange(omega_bounds,n,target_t,model_kwargs,init_cond)
        sample_ls_PCE.append([])
        time_ls_PCE.append([])
        for m in M_ls:
            sample_ls_PCE[i].append(evaluate_pce(interpolator,omega_dist,m))
            time_ls_PCE[i].append(time.time()-start_time_PCE)
    end_time_PCE = time.time()
    sample_ls_PCE = np.asarray(sample_ls_PCE,dtype=list)
    time_ls_PCE = np.asarray(time_ls_PCE)
    # ====================================================================
    # Monte Carlo Sampling
    # ====================================================================
    sample_ls_MC = []
    time_ls_MC = []

    start_time_PCE = time.time()
    for m in M_ls:
        sample_ls_MC.append(estimate_monte_carlo(omega_dist,m,target_t,model_kwargs,init_cond))
        time_ls_MC.append(time.time()-start_time_PCE)
    end_time_PCE = time.time()

    sample_ls_MC = np.asarray(sample_ls_MC, dtype=list)
    time_ls_MC = np.asarray(time_ls_MC)
    # ====================================================================
    # Compute errors
    # ====================================================================
    mean_err_PCE = np.zeros((len(N_ls),len(M_ls)))
    var_err_PCE = np.zeros((len(N_ls),len(M_ls)))

    for i in range(len(N_ls)):
        for j in range(len(M_ls)):
            mean_err_PCE[i,j],var_err_PCE[i,j] = compute_errors(sample_ls_PCE[i,j],ref_mean,ref_var)
    
    mean_err_MC = np.zeros(len(M_ls))
    var_err_MC = np.zeros(len(M_ls))

    for j in range(len(M_ls)):
        mean_err_MC[j], var_err_MC[j] = compute_errors(sample_ls_MC[j],ref_mean,ref_var) 
    # ====================================================================
    # Plotting.
    # ====================================================================
    fig, axs = plt.subplots(2,2)
    pce_mean_plot_ls = []
    pce_var_plot_ls = []
    pce_time_plot_ls = []
    # Plot PCE errors/times with one graph per node count
    for i in range(len(N_ls)):
        mean_plt = axs[0,0].plot(M_ls,mean_err_PCE[i], c=plt.colormaps['cool'](i/len(N_ls)))
        var_plt = axs[0,1].plot(M_ls,var_err_PCE[i], c=plt.colormaps['cool'](i/len(N_ls)))
        time_plt = axs[1,1].plot(M_ls,time_ls_PCE[i], c=plt.colormaps['cool'](i/len(N_ls)))
        pce_time_plot_ls.append(time_plt[0])
        pce_var_plot_ls.append(var_plt[0])
        pce_mean_plot_ls.append(mean_plt[0])


    # Plot Monte Carlo errors/times
    mc_mean_plt = axs[0,0].plot(M_ls,mean_err_MC,c='red')[0]
    mc_var_plt = axs[0,1].plot(M_ls,var_err_MC,c='red')[0]
    mc_time_plt = axs[1,0].plot(M_ls,time_ls_MC,c='red')[0]

    axs[0,0].set_title(r"Relative errors on $\mu$ estimates")
    axs[0,1].set_title(r"Relative errors on variance estimates")
    axs[1,0].set_title(r"Compute time Monte Carlo")
    axs[1,1].set_title(r"Compute time Lagrange Interpolation")
    fig.suptitle("Comparison Lagrange Interpolation with increasing nodes vs Monte Carlo")

    axs[0,0].set_xlabel(r"Number of samples [log]")
    axs[0,1].set_xlabel(r"Number of samples [log]")    
    axs[1,0].set_xlabel(r"Number of samples")
    axs[1,1].set_xlabel(r"Number of samples")

    axs[0,0].set_ylabel(r"Relative error [log]")
    axs[0,1].set_ylabel(r"Relative error [log]")    
    axs[1,0].set_ylabel(r"Time [s]")
    axs[1,1].set_ylabel(r"Time [s]")

    fig.legend(pce_mean_plot_ls+[mc_mean_plt] , [f"Nodes:{n}" for n in N_ls] + ["Monte Carlo"])

    axs[0,0].set_xscale("log")
    axs[0,1].set_xscale("log")
    
    axs[0,0].set_yscale("log")
    axs[0,1].set_yscale("log")

    plt.tight_layout()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    
    # ====================================================================