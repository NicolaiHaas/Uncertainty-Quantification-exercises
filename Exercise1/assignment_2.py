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
    samples = a + (b - a) * samples
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
    # TODO: compute the integral with the Monte Carlo method.
    # Depending on 'with_transform', use the uniform distribution on [a, b]
    # directly or transform the uniform distribution on [0, 1] to [a, b].
    # Return the integral estimate and the corresponding RMSE.
    # ====================================================================
    if with_transform:
        mean, rmse = monte_carlo(p=cp.Uniform(0.0, 1.0),
            n_samples=n_samples, f=f, 
            transform=lambda samples: transform(samples, a, b),
            seed=seed
        )

    else:
        mean, rmse = monte_carlo(p=cp.Uniform(a, b),
            n_samples=n_samples, f=f, seed=seed
        )


    integral = (b - a) * mean[0]
    rmse = (b - a) * rmse[0]

    # ====================================================================
    return integral, rmse


if __name__ == "__main__":
    # TODO: define the parameters of the simulation.
    # ====================================================================
    sample_sizes = [10, 100, 1000, 10000]

    sim_label = "Assignment 2.1: direct sampling on [0, 1]"
    a = 0.0
    b = 1.0
    with_transform = False
    plot_fname = "plots/assignment_2_1_direct_0_1.png"

    # ====================================================================

    # TODO: compute the integral and the errors.
    # ====================================================================
    
    # helper funs for each sim
    def run_mc_for_sample_sizes(a, b, sample_sizes, with_transform):
        true_val = analytical_integral(a, b)

        estimates = []
        errors = []
        rmses = []

        for n_samples in sample_sizes:
            estimate, rmse = integrate_mc(f, a, b, n_samples, with_transform)
            error = abs(true_val - estimate)

            estimates.append(estimate)
            errors.append(error)
            rmses.append(rmse)

            print(f"N = {n_samples:5d} estimate = {estimate:.8f} error = {error:.8f} RMSE = {rmse:.8f}")

        return np.array(estimates), np.array(errors), np.array(rmses)
        
   


    print(sim_label)
    estimates, errors, rmses = run_mc_for_sample_sizes(a, b, sample_sizes, with_transform)

    # ====================================================================

    # TODO: plot the results on the log-log scale.
    # ====================================================================
    # Compute global limits so all plots use identical axes and are
    # directly comparable.

    def plot_mc_result(label, sample_sizes, errors, rmses, filename):
        plt.figure(figsize=(7, 5))

        plt.plot(sample_sizes, errors, marker="o", label="Absolute error")
        plt.plot(sample_sizes, rmses, marker="o", linestyle="--", label="RMSE")

        plt.xscale("log")
        plt.yscale("log")

        plt.xlabel("Number of samples N")
        plt.ylabel("Error")
        plt.title(label)
        plt.grid(True, which="both")
        plt.legend()

        plt.tight_layout()
        plt.savefig(filename, bbox_inches="tight")
        plt.show()


    plot_mc_result(sim_label, sample_sizes, errors, rmses, plot_fname)


    # estimates, errors, rmses = run_mc_for_sample_sizes(2.0, 4.0, sample_sizes, False)
    # plot_mc_result("Assignment 2.2: direct sampling on [2, 4]", sample_sizes, errors, rmses, "plots/assignment_2_2_direct_2_4.png")
    # estimates, errors, rmses = run_mc_for_sample_sizes(2.0, 4.0, sample_sizes, True)
    # plot_mc_result("Assignment 2.2: transformed sampling on [2, 4]", sample_sizes, errors, rmses, "plots/assignment_2_2_transformed_2_4.png")
    # ====================================================================
    