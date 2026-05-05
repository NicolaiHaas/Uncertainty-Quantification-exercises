import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.sampling import control_variates, importance_sampling, monte_carlo


def f(x: npt.NDArray) -> npt.NDArray:
    # TODO: define the target function.
    # ====================================================================
    return np.exp(x)
    # ====================================================================


def analytical_integral() -> float:
    # TODO: compute the analytical integral of f on [0, 1].
    # ====================================================================
    return np.e-1
    # ====================================================================


def run_monte_carlo(Ns: tuple[int, ...], seed: int = 42) -> list[float]:
    # TODO: run the Monte Carlo method and return the absolute error
    # of the estimation.
    # ====================================================================
    np.random.seed(seed=seed)
    errors = []
    for n in Ns:
        samples = np.random.random(size=n)
        errors.append(
                np.abs(np.mean(f(samples)) - analytical_integral())
            )
    return errors
    # ====================================================================


def run_control_variates(
    Ns: tuple[int, ...], seed: int = 42
):
    # TODO: run the control variate method for and return the absolute
    # errors of the resulting estimations.
    # ====================================================================
    h_1 = lambda x: x
    h_2 = lambda x: 1 + x
    h_3 = lambda x: 1 + x + x**2/2
    # expected values and variances under the transformations
    E_h = [0.5,1.5,5/3]
    V_h = [1/12,1/12,17/90]
    #list of transformations
    H = [h_1,h_2,h_3]

    np.random.seed(seed=seed)
    errors = [[] for _ in range(len(H))]
    for n in Ns:
        samples = np.random.random(n)
        f_samples = f(samples)
        for j in range(len(H)):
            h_samples = H[j](samples)
            a = np.cov(f_samples,h_samples)[0,1]/V_h[j]
            errors[j].append(
                    np.abs(np.mean(f(samples)) + a*(E_h[j]-np.mean(h_samples)) - analytical_integral())
                )
    return tuple(errors)
    # ====================================================================


def run_importance_sampling(
    Ns: tuple[int, ...], seed: int = 42
):
    # TODO: run the importance sampling method and return the absolute
    # errors of the resulting estimations.
    # ====================================================================
    q_1 = cp.Beta(5,1,0,1).pdf
    q_2 = cp.Beta(0.5,0.5,0,1).pdf
    Q = [q_1,q_2]
    np.random.seed(seed=seed)
    errors = [[] for _ in range(len(Q))]
    for n in Ns:
        x = np.random.rand(n)
        for j in range(len(Q)):
            # p is uniform on [0,1] => w = 1/q
            errors[j].append(
                np.abs(np.mean(f(x)/Q[j](x)) - analytical_integral())
            )
        
    return tuple(errors)
    # ====================================================================


if __name__ == "__main__":
    # TODO: define the parameters of the simulation.
    # ====================================================================
    N = (10,100,1000,10000)
    # ====================================================================

    # TODO: run all the methods.
    # ====================================================================
    MC_errors = run_monte_carlo(N)
    CV_errors = run_control_variates(N)
    IS_errors = run_importance_sampling(N)
    # ====================================================================

    # TODO: plot the results on the log-log scale.
    # ====================================================================
    MC = plt.plot(N,MC_errors, c = 'red', ls = '--')[0]
    CV_0 = plt.plot(N,CV_errors[0])[0]
    CV_1 = plt.plot(N,CV_errors[1])[0]
    CV_2 = plt.plot(N,CV_errors[2])[0]
    IS_0 = plt.plot(N,IS_errors[0])[0]
    IS_1 = plt.plot(N,IS_errors[1])[0]
    
    plt.yscale('log')
    plt.xscale('log')

    plt.legend([MC,CV_0,CV_1,CV_2,IS_0,IS_1],["MC","CV x","CV x+1","CV x^2+x+1","IS a=5,b=1","IS a=b=0.5"])
    plt.show()
    # ====================================================================
