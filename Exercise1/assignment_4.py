from collections import defaultdict

import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.oscillator import Oscillator


def load_reference(filename: str) -> tuple[float, float]:
    # TODO: load reference values for the mean and variance.
    # ====================================================================
    with open(filename,'r') as f:
        mean = float(f.readline())
        var = float(f.readline())
    # ====================================================================
    return mean, var


def simulate(
    t_grid: npt.NDArray,
    omega_distr: cp.Distribution,
    n_samples: int,
    model_kwargs: dict[str, float],
    init_cond: dict[str, float],
    rule="random",
    seed=42,
) -> npt.NDArray:
    # TODO: simulate the oscillator with the given parameters and return
    # generated solutions.
    # ====================================================================
    c = model_kwargs.get("c",1)
    k = model_kwargs.get('k',1)
    f = model_kwargs.get('f',1)
    omegas = omega_distr.sample(n_samples,seed=seed,rule=rule)
    mth = init_cond.get('method','euler')
    y0 = init_cond.get('y0',0)
    y1 = init_cond.get('y1',0)


    sample_solutions = [
                    Oscillator(c,k,f,omega).discretize(mth,y0,y1,t_grid) 
                    for omega in omegas
                 ] 
    # ====================================================================
    return sample_solutions


def compute_errors(
    samples: npt.NDArray, mean_ref: float, var_ref: float
) -> tuple[float, float]:
    # TODO: compute the relative errors of the mean and variance
    # estimates.
    # ====================================================================
    sample_mean = np.mean(np.array(samples)[:,-1])
    sample_var = np.var(np.array(samples)[:,-1],ddof=1)
    mean_error, var_error = abs(1-sample_mean/mean_ref), abs(1-sample_var/var_ref)
    # ====================================================================
    return float(mean_error), float(var_error)


if __name__ == "__main__":
    # TODO: define the parameters of the simulations.
    # ====================================================================
    c = 0.5
    k = 2
    f = 0.5
    model = {'c':c,'k':k,'f':f}
    y0 = 0.5
    y1 = 0
    method = 'euler'
    init = {'y0':y0,'y1':y1,'method':method}
    t_grid = np.linspace(0,10,int(10/0.01))
    omega = cp.Uniform(0.95,1.05)
    N = [10, 100, 1000, 10000]
    # ====================================================================

    # TODO: run the simulations.
    # ====================================================================
    
    ref_mean, ref_var = load_reference("./data/oscillator_ref.txt")
    m_errors_r = []
    v_errors_r = []
    m_errors_h = []
    v_errors_h = []
    traj_ls = []
    for n in N:
        samples_r = simulate(t_grid,omega,n,model,init,'random')
        samples_h = simulate(t_grid,omega,n,model,init,'halton')
        m_err_r, v_err_r = compute_errors(samples_r,ref_mean,ref_var)
        m_err_h, v_err_h = compute_errors(samples_h,ref_mean,ref_var)
        m_errors_r.append(m_err_r)
        v_errors_r.append(v_err_r)
        m_errors_h.append(m_err_h)
        v_errors_h.append(v_err_h)
    # ====================================================================

    # TODO: compute the statistics.
    # ====================================================================
    # ====================================================================

    # TODO: plot the results on the log-log scale.
    # ====================================================================
    r_m = plt.plot(N,m_errors_r, ls = '--')[0]
    r_v = plt.plot(N,v_errors_r, ls = '--')[0]
    h_m = plt.plot(N,m_errors_h)[0]
    h_v = plt.plot(N,m_errors_h)[0]

    plt.legend([r_m,r_v,h_m,h_v],["Random µ error", "Random Variance error", "Halton µ error","Halton variance error"])
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('error')
    plt.xlabel('#samples')
    plt.show()
    # ====================================================================

    # TODO: plot sampled trajectories.
    # ====================================================================
    curves = simulate(t_grid,omega,10,model,init,'random')
    for curve in curves:
        plt.plot(t_grid,curve) 
    plt.xlabel('amplitude')
    plt.ylabel("time")
    plt.show()
    # ====================================================================
