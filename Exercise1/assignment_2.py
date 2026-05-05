from functools import partial
from typing import Callable

import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.sampling import monte_carlo

Function = Callable[[npt.NDArray], npt.NDArray]


def f(x: npt.NDArray) -> npt.NDArray:
    # TODO: define the target function.
    # ====================================================================
    return np.sin(x)
    # ====================================================================


def analytical_integral(a: float, b: float) -> float:
    # TODO: compute the analytical integral of f on [a, b].
    # ====================================================================
    return np.cos(a) - np.cos(b)
    # ====================================================================


def transform(samples: npt.NDArray, a: float, b: float) -> npt.NDArray:
    # TODO: implement the transformation of U from [0, 1] to [a, b].
    # ====================================================================
    samples = samples * (b-a) + a
    # ====================================================================
    return samples


def integrate_mc(
    f: Function,
    a: float,
    b: float,
    n_samples: int,
    with_transform: bool = False,
    seed: int = 42,
) -> tuple[float, float]:
    # TODO: compute the integral with the Monta Carlo method.
    # Depending on 'with_transform', use the uniform distribution on [a, b]
    # directly or transform the uniform distribution on [0, 1] to [a, b].
    # Return the integral estimate and the corresponding RMSE.
    # ====================================================================
    if with_transform:
        samples = np.random.uniform(a,b,n_samples)
    else:
        samples = transform(np.random.rand(n_samples),a,b)
    length = b-a
    integral = float(np.mean(f(samples))*length)
    rmse = float(np.std(samples,ddof=1)/np.sqrt(n_samples))
    # ====================================================================
    return integral, rmse


if __name__ == "__main__":
    # TODO: define the parameters of the simulation.
    # ====================================================================
    N = [10,100,1000,10000]
    a = 2 #0
    b = 4 #1
    # ====================================================================

    # TODO: compute the integral and the errors.
    # ====================================================================
    eps_ls = []
    I_ls = []
    rmse_ls = []
    for n in N:
        I,rmse = integrate_mc(f,a,b,n)
        eps_ls.append(abs(I-analytical_integral(a,b)))
        I_ls.append(I)
        rmse_ls.append(rmse)
    # ====================================================================

    # TODO: plot the results on the log-log scale.
    # ====================================================================
    I_plot = plt.plot(N,I_ls, c = 'blue')[0]
    rmse_plot = plt.plot(N,rmse_ls, c = 'red')[0]
    eps_plot = plt.plot(N,eps_ls, c = 'green')[0]

    plt.xlabel("Number of samples")
    plt.xscale('log')
    plt.yscale('log')

    plt.legend([I_plot,rmse_plot,eps_plot],["Approximated Integral","RMSE", r"$\epsilon$"])
    plt.show()
    # ====================================================================
