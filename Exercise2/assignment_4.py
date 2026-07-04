import chaospy as cp
import matplotlib.pyplot as plt
import numpoly as npoly
import numpy as np
import numpy.typing as npt

from utils.helpers import load_reference, simulate

dt = .1 # As in exercise 1 time step of ODE simulation

'''
    ...
    Args:
        nodes:
        weights:
        polynomials: numpoly.ndpoly     - An array of 2 dimensional polynomials (of time and omega) up to degree N,
                                          depending on the polynomial desired for quadrature. 
                                          Omega (o) is argument 0, Time (t) is argument 1.
'''
def compute_coefficients(
    nodes: npt.NDArray,
    weights: npt.NDArray,
    polynomials: npoly.ndpoly,
    target_t: float,
    model_kwargs: dict,
    init_cond: dict,
    mode: str,
) -> npt.NDArray:
    # TODO: compute the coefficients using the quadrature rule, either manually.
    # or using chaospy functionality.
    # ====================================================================
    assert mode in ["manual","cp", "chaospy"], "Mode for coefficient computation must be either chaospy or manual"

    t_grid = np.arange(0,target_t+dt,dt)
    # function values at nodes
    node_solutions = simulate(t_grid, nodes, model_kwargs, init_cond)[:,-1]

    # It appears we are already given nodes and weights so no real quadrature computation
    if mode == "manual":
        coefficients = polynomials(nodes)@np.multiply(node_solutions,weights)
    else:
        _, coefficients = cp.fit_quadrature(polynomials,nodes,weights,node_solutions,retall=1)
    # ====================================================================
    return coefficients


def compute_moments(coefficients: npt.NDArray) -> tuple[float, float]:
    # TODO: compute the target mean and variance from the PCE coefficients.
    # ===================================================================
    mean, variance = coefficients[0], np.sum(coefficients[1:]**2)
    # ====================================================================
    return mean, variance


if __name__ == "__main__":
    # Define the parameters of the simulations.
    # ====================================================================
    np.random.seed(42)
    
    mc_samples = 1000000
    N_ls = np.arange(6) + 1
    c = 0.5
    k = 2
    f = 0.5

    model_kwargs = {
        "c":c,
        "k":k,
        "f":f
    }

    y0 = 0.5
    y1 = 0

    init_cond ={
        "y0":y0,
        "y1":y1
    }

    omega_dist = cp.Uniform(0.95,1.05)
    target_t = 10

    man_mean_errors = []
    man_var_errors  = []
    cp_mean_errors = []
    cp_var_errors = []

    ref_mean, ref_var = load_reference("oscillator_ref.txt")
    # ====================================================================
    # Compute pseudo-spectral coefficients.
    # ====================================================================
    for N in N_ls:
        polys_N = cp.generate_expansion(N,omega_dist,normed=True)
        nodes_N, weights_N = cp.generate_quadrature(N,omega_dist)
        
        # Make sure no weird formatting from generate expansion / ndim == 1 
        nodes_N = nodes_N.reshape(-1)
        weights_N = weights_N.reshape(-1)

        man_coeff = compute_coefficients(nodes_N,weights_N,polys_N,target_t,model_kwargs,init_cond,mode="manual")
        cp_coeff = compute_coefficients(nodes_N,weights_N,polys_N,target_t,model_kwargs,init_cond,mode="cp")
    # ====================================================================
    # Compute the momements and calcualte their errors. (Still in for loop)
    # ====================================================================
        man_mean, man_var = compute_moments(man_coeff)
        man_mean_errors.append(
            np.abs((man_mean-ref_mean)/ref_mean)
        )
        man_var_errors.append(
            np.abs((man_var-ref_var)/ref_var)
        )
        cp_mean, cp_var = compute_moments(cp_coeff)
        cp_mean_errors.append(
            np.abs((cp_mean-ref_mean)/ref_mean)
        )
        cp_var_errors.append(
            np.abs((cp_var-ref_var)/ref_var)
        )
    # ====================================================================
    # Reference MC solution
    # ====================================================================
    t_grid = np.arange(0,target_t+dt,dt)
    samples = simulate(
        t_grid,
        omega_dist.sample(mc_samples),
        model_kwargs,
        init_cond,
        progress = True
        )[:,-1]
    mc_mean = samples.mean()
    mc_var = samples.var(ddof=1,mean=mc_mean)

    mc_mean_err = np.abs((mc_mean-ref_mean)/ref_mean)
    mc_var_err = np.abs((mc_var-ref_var)/ref_var)
    # ====================================================================
    # Plot the results.
    # ====================================================================
    fig, axs = plt.subplots(1,2)
    cp_mean_plot = axs[0].plot(N_ls,cp_mean_errors, c = "purple", ls = 'dashed')[0]
    man_mean_plot = axs[0].plot(N_ls,man_mean_errors, c = "yellow",ls=(0,(1,1)))[0]
    mc_mean_plot = axs[0].plot(N_ls,np.zeros_like(N_ls)+mc_mean_err,c="green")[0]

    cp_var_plot = axs[1].plot(N_ls,cp_var_errors, c = "purple")[0]
    man_var_plot = axs[1].plot(N_ls,man_var_errors, c = "yellow")[0]
    mc_var_plot = axs[1].plot(N_ls,np.zeros_like(N_ls)+mc_var_err,c="green")[0]
    plt.setp(axs, yscale = 'log', xlabel = "Size of polynomial basis", ylabel = "Error")
    
    axs[0].set_title("Relative mean error")
    axs[1].set_title("Relative variance error")
    plt.suptitle("Calculation of spectral coefficients manual vs. Chaospy vs. MC baseline")
    fig.legend([cp_var_plot,man_var_plot,mc_var_plot],["Chospy","Manual",fr"$MC_{{N={mc_samples}}}$"])
    plt.show()
    # ====================================================================
