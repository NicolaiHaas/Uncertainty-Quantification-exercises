import time

import chaospy as cp
import numpy as np

import matplotlib.pyplot as plt

from utils.sobol import monte_carlo_sobol, pseudo_spectral_sobol


def get_distribution(
    c_lims: tuple[float, float],
    k_lims: tuple[float, float],
    f_lims: tuple[float, float],
    y0_lims: tuple[float, float],
    y1_lims: tuple[float, float],
) -> cp.Distribution:
    """Creates the joint distribution over the stochastic parameters."""

    return cp.J(
        cp.Uniform(*c_lims), # c
        cp.Uniform(*k_lims), # k
        cp.Uniform(*f_lims), # f
        cp.Uniform(*y0_lims), # y0
        cp.Uniform(*y1_lims) # y1
    )


def run_method(method, **kwargs):
    """Runs the specified method and prints the results.

    The results include the first and total order Sobol' indices as well as
    the elapsed time to run the method."""

    joint = kwargs["distribution"]
    t_grid = kwargs["t_grid"]
    fixed_args = kwargs["fixed_args"]

    pce_degrees = kwargs.get("pce_degrees", [])
    quadrature_degrees = kwargs.get("quadrature_degrees", [])

    n_samples = kwargs.get("n_samples", 0)

    if method == "full_PCE":
        res = []
        input_times = []
        start_t = time.time()
        for pce_degree in pce_degrees:
            for quadrature_degree in quadrature_degrees:
                t0 = time.time()
                res.append(
                    pseudo_spectral_sobol(
                        pce_degree,
                        quadrature_degree,
                        joint,
                        t_grid,
                        fixed_args,
                        sparse=False,
                    )
                )
                input_times.append(time.time() - t0)
        res = np.asarray(res)
        end_t = time.time()
    elif method == "sparse_PCE":
        res = []
        input_times = []
        start_t = time.time()
        for pce_degree in pce_degrees:
            for quadrature_degree in quadrature_degrees:
                t0 = time.time()
                res.append(
                    pseudo_spectral_sobol(
                        pce_degree,
                        quadrature_degree,
                        joint,
                        t_grid,
                        fixed_args,
                    )
                )
                input_times.append(time.time() - t0)
        res = np.asarray(res)
        end_t = time.time()
    else:
        start_t = time.time()
        res = monte_carlo_sobol(n_samples, joint, t_grid, fixed_args)
        end_t = time.time()
        input_times = [end_t - start_t]
    
    return res, end_t - start_t, input_times


if __name__ == "__main__":
    # TODO: set the stochastic parameters.
    c_lims = (0.08,0.12)
    k_lims = (0.03,0.04)
    f_lims = (0.08,0.12)
    y0_lims = (0.45,0.55)
    y1_lims =  (-0.05,0.05)

    # TODO: set the determinisic parameters.
    fixed_args = {"omega": 1.0}

    # TODO: set the parameters of the methods.
    quadrature_degrees = [3,4]
    pce_degrees = [3,4]
    n_samples = 446

    # TODO: set the time domain
    T_max = 10
    dt = 0.01
    t_grid = np.arange(0, T_max + dt, dt, dtype=np.float64)

    ###########################################################################

    joint = get_distribution(c_lims,k_lims,f_lims,y0_lims,y1_lims)

    full_S_PCE, t_full_PCE, full_input_times = run_method(
        "full_PCE",
        distribution = joint,
        t_grid = t_grid,
        fixed_args = fixed_args,
        pce_degrees = pce_degrees,
        quadrature_degrees = quadrature_degrees
    )

    sparse_S_PCE, t_sparse_PCE, sparse_input_times = run_method(
        "sparse_PCE",
        distribution = joint,
        t_grid = t_grid,
        fixed_args = fixed_args,
        pce_degrees = pce_degrees,
        quadrature_degrees = quadrature_degrees
    )

    mc_res, t_MC, mc_input_times = run_method(
        "mc",
        distribution = joint,
        t_grid = t_grid,
        fixed_args = fixed_args,
        n_samples = n_samples
    )
    mc_S, mc_S_T = mc_res


    
    fig, axs = plt.subplots(1,2)
    sing_handle = []
    tot_handle = []
    for i,run in enumerate(full_S_PCE):
        sing_handle += axs[0].plot(run[0], c=plt.colormaps['Blues'](i/4 + 0.2))  # singular Sobols
        tot_handle += axs[1].plot(run[1], c=plt.colormaps['Reds'](i/4 + 0.2))   # total Sobols
    
    for i,run in enumerate(sparse_S_PCE):
        sparse_sing_handle = axs[0].plot(run[0], c=plt.colormaps['Blues'](i/4 + 0.2), ls = 'dotted')[0]  # sparse singular Sobols
        sparse_tot_handle = axs[1].plot(run[1], c=plt.colormaps['Reds'](i/4 + 0.2), ls = 'dotted')[0]   # sparse total Sobols
    
    mc_sing_handle = axs[0].plot(mc_S, c = 'midnightblue', ls = 'dashed')
    mc_tot_handle = axs[1].plot(mc_S_T, c = 'darkred', ls = 'dashed')
    plt.setp(
        axs,
        #yscale = 'log', 
        xlabel = 'Parameter', ylabel='Sobol index', 
        xticks = range(5),xticklabels=[r"$c$",r"$k$",r"$f$",r"$y_0$",r"$y_1$"]
    )

    axs[0].legend(
        sing_handle + mc_sing_handle,
        ["K=3, N=3","K=4,N=3","K=3,N=4","K=4,N=4","MC"]
    )
    axs[1].legend(
        tot_handle + mc_tot_handle,
        ["K=3, N=3","K=4,N=3","K=3,N=4","K=4,N=4","MC"]
    )
    plt.show()

    full_times_tag = plt.plot(full_input_times)
    sparse_time_tag = plt.plot(sparse_input_times)
    mc_time_tag = plt.plot([t_MC]*len(sparse_input_times))

    plt.xticks(range(4),["(3,3)","(4,3)","(3,4)","(4,4)"])
    plt.xlabel("Configuration (K,N)")
    plt.ylabel("Computation time")
    plt.legend(full_times_tag + sparse_time_tag + mc_time_tag,["Full Grid PCE","Sparse Grid PCE", "MC"])
    plt.show()
    ###########################################################################
