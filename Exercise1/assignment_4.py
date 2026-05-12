from collections import defaultdict

import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.oscillator import Oscillator


def load_reference(filename: str) -> tuple[float, float]:
    # TODO: load reference values for the mean and variance.
    # ====================================================================
    reference_values = np.loadtxt(filename, dtype=float)

    mean = reference_values[0]
    var = reference_values[1]
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
    omega_samples = omega_distr.sample(
        size=n_samples,
        rule=rule,
        seed=seed
    )

    omega_samples = np.asarray(omega_samples, dtype=float)# .reshape(-1)

    sample_solutions = np.zeros((n_samples, len(t_grid)))

    for sample_idx, omega in enumerate(omega_samples):
        sample_solutions[sample_idx] = Oscillator(**model_kwargs, omega=float(omega)).discretize(
            method="odeint",
            y0=init_cond["y0"],
            y1=init_cond["y1"],
            t_grid=t_grid
        )
    # ====================================================================
    return sample_solutions


def compute_errors(
    samples: npt.NDArray, mean_ref: float, var_ref: float
) -> tuple[float, float]:
    # TODO: compute the relative errors of the mean and variance
    # estimates.
    # ====================================================================
    # last mat col
    final_positions = samples[:, -1]

    mean_approx = np.mean(final_positions)
    var_approx = np.var(final_positions, ddof=1)

    mean_error = abs(1.0 - mean_approx / mean_ref)
    var_error = abs(1.0 - var_approx / var_ref)
    # ====================================================================
    return mean_error, var_error


if __name__ == "__main__":
    # TODO: define the parameters of the simulations.
    # ====================================================================
    sample_sizes = [10, 100, 1000, 10000]

    dt = 0.01
    t_min = 0.0
    t_max = 10.0
    t_grid = np.arange(t_min, t_max + dt, dt)

    model_kwargs = {
        "c": 0.5,
        "k": 2.0,
        "f": 0.5,
    }

    init_cond = {
        "y0": 0.5,
        "y1": 0.0,
    }

    omega_distr = cp.Uniform(0.95, 1.05)

    mean_ref, var_ref = load_reference("data/oscillator_ref.txt")

    deterministic_oscillator = Oscillator(**model_kwargs, omega=1.0)

    deterministic_solution = deterministic_oscillator.discretize(
        method="odeint",
        y0=init_cond["y0"],
        y1=init_cond["y1"],
        t_grid=t_grid,
    )
    # ====================================================================

    # TODO: run the simulations.
    # ====================================================================

    # normal mc
    mc_sols = [simulate(t_grid=t_grid, omega_distr=omega_distr, n_samples=n_samples,
                    model_kwargs=model_kwargs, init_cond=init_cond, rule="random")
                     for n_samples in sample_sizes]
    # quasi mc
    qmc_sols = [simulate(t_grid=t_grid, omega_distr=omega_distr, n_samples=n_samples,
                    model_kwargs=model_kwargs, init_cond=init_cond, rule="halton")
                     for n_samples in sample_sizes]
    # ====================================================================

    # TODO: compute the statistics.
    # ====================================================================
        
    mean_errors_mc = []
    var_errors_mc = []

    for sample_solutions in mc_sols:
        mean_error, var_error = compute_errors(
            samples=sample_solutions,
            mean_ref=mean_ref,
            var_ref=var_ref,
        )

        mean_errors_mc.append(mean_error)
        var_errors_mc.append(var_error)


    mean_errors_qmc = []
    var_errors_qmc = []

    for sample_solutions in qmc_sols:
        mean_error, var_error = compute_errors(
            samples=sample_solutions,
            mean_ref=mean_ref,
            var_ref=var_ref,
        )

        mean_errors_qmc.append(mean_error)
        var_errors_qmc.append(var_error)

    print("Assignment 4: relative errors for y(10)\n")
    print(f"Reference mean:      {mean_ref:.8f}")
    print(f"Reference variance:  {var_ref:.8f}")
    print(f"Deterministic y(10): {deterministic_solution[-1]:.8f}\n")
    print("Monte Carlo mean errors:")
    print(mean_errors_mc)
    print("Monte Carlo variance errors:")
    print(var_errors_mc)
    print("Quasi-Monte Carlo mean errors:")
    print(mean_errors_qmc)
    print("Quasi-Monte Carlo variance errors:")
    print(var_errors_qmc)
    # ====================================================================

    # TODO: plot the results on the log-log scale.
    # ====================================================================
    plt.figure(figsize=(8, 5))

    plt.plot(sample_sizes, mean_errors_mc, marker="o", label="Monte Carlo: mean")
    plt.plot(sample_sizes, var_errors_mc, marker="o", linestyle="--", label="Monte Carlo: variance")

    plt.plot(sample_sizes, mean_errors_qmc, marker="o", label="Quasi-Monte Carlo: mean")
    plt.plot(sample_sizes, var_errors_qmc, marker="o", linestyle="--", label="Quasi-Monte Carlo: variance")

    # reference mc convergence rate
    plt.plot(sample_sizes, 1 / np.sqrt(np.array(sample_sizes)), 
        linestyle=":", color="black", label=r"$\mathcal{O}(N^{-1/2})$")

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Number of samples N")
    plt.ylabel("Relative error")
    plt.title("Forward uncertainty propagation for y(10)")
    plt.grid(True, which="both")
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/assignment_4_relative_errors.png", bbox_inches="tight")
    # ====================================================================

    # TODO: plot sampled trajectories.
    # ====================================================================
    trajectory_samples = mc_sols[0][:10]

    plt.figure(figsize=(8, 5))

    for sample_solution in trajectory_samples:
        plt.plot(t_grid, sample_solution, alpha=0.8)

    plt.plot(
        t_grid,
        deterministic_solution,
        linestyle="--",
        linewidth=2,
        label="Deterministic solution",
    )

    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.title("Ten oscillator trajectories for random omega samples")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/assignment_4_trajectories.png", bbox_inches="tight")

    plt.show()
    # ====================================================================