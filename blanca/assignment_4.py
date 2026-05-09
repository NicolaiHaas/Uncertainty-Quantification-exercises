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

    mean = float(reference_values[0])
    var = float(reference_values[1])
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
        seed=seed,
    )

    omega_samples = np.asarray(omega_samples, dtype=float).reshape(-1)

    sample_solutions = np.zeros((n_samples, len(t_grid)))

    for sample_idx, omega in enumerate(omega_samples):
        oscillator = Oscillator(
            **model_kwargs,
            omega=float(omega),
        )

        sample_solutions[sample_idx] = oscillator.discretize(
            method="odeint",
            y0=init_cond["y0"],
            y1=init_cond["y1"],
            t_grid=t_grid,
        )
    # ====================================================================
    return sample_solutions


def compute_errors(
    samples: npt.NDArray, mean_ref: float, var_ref: float
) -> tuple[float, float]:
    # TODO: compute the relative errors of the mean and variance
    # estimates.
    # ====================================================================
    final_positions = samples[:, -1]

    mean_approx = np.mean(final_positions)

    # We use the unbiased sample variance estimator, consistently with the
    # previous assignments.
    var_approx = np.var(final_positions, ddof=1)

    mean_error = abs(1.0 - mean_approx / mean_ref)
    var_error = abs(1.0 - var_approx / var_ref)
    # ====================================================================
    return mean_error, var_error


if __name__ == "__main__":
    # TODO: define the parameters of the simulations.
    # ====================================================================
    Ns = [10, 100, 1000, 10000]

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

    deterministic_oscillator = Oscillator(
        **model_kwargs,
        omega=1.0,
    )

    deterministic_solution = deterministic_oscillator.discretize(
        method="odeint",
        y0=init_cond["y0"],
        y1=init_cond["y1"],
        t_grid=t_grid,
    )
    # ====================================================================

    # TODO: run the simulations.
    # ====================================================================
    methods = {
        "Monte Carlo": "random",
        "Quasi-Monte Carlo": "halton",
    }

    solutions = defaultdict(list)

    for method_name, rule in methods.items():
        for n_samples in Ns:
            sample_solutions = simulate(
                t_grid=t_grid,
                omega_distr=omega_distr,
                n_samples=n_samples,
                model_kwargs=model_kwargs,
                init_cond=init_cond,
                rule=rule,
            )

            solutions[method_name].append(sample_solutions)
    # ====================================================================

    # TODO: compute the statistics.
    # ====================================================================
    mean_errors = defaultdict(list)
    var_errors = defaultdict(list)

    print("Assignment 4: relative errors for y(10)\n")
    print(f"Reference mean:     {mean_ref:.8f}")
    print(f"Reference variance: {var_ref:.8f}")
    print(f"Deterministic y(10): {deterministic_solution[-1]:.8f}\n")

    print(
        f"{'Method':>18} | {'N':>8} | {'Mean error':>12} | "
        f"{'Variance error':>14}"
    )

    for method_name in methods:
        for n_samples, sample_solutions in zip(Ns, solutions[method_name]):
            mean_error, var_error = compute_errors(
                samples=sample_solutions,
                mean_ref=mean_ref,
                var_ref=var_ref,
            )

            mean_errors[method_name].append(mean_error)
            var_errors[method_name].append(var_error)

            print(
                f"{method_name:>18} | "
                f"{n_samples:8d} | "
                f"{mean_error:12.6e} | "
                f"{var_error:14.6e}"
            )
    # ====================================================================

    # TODO: plot the results on the log-log scale.
    # ====================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for method_name in methods:
        ax.loglog(
            Ns,
            mean_errors[method_name],
            marker="o",
            label=f"{method_name}: mean",
        )

        ax.loglog(
            Ns,
            var_errors[method_name],
            marker="o",
            linestyle="--",
            label=f"{method_name}: variance",
        )

    # Reference Monte Carlo convergence rate O(N^{-1/2}).
    ax.loglog(
        Ns,
        1 / np.sqrt(np.array(Ns)),
        linestyle=":",
        color="black",
        label=r"$\mathcal{O}(N^{-1/2})$",
    )

    ax.set_xlabel("Number of samples N")
    ax.set_ylabel("Relative error")
    ax.set_title("Forward uncertainty propagation for y(10)")
    ax.grid(True, which="both")
    ax.legend()

    fig.tight_layout()
    fig.savefig("assignment_4_relative_errors.png", bbox_inches="tight")
    # ====================================================================

    # TODO: plot sampled trajectories.
    # ====================================================================
    trajectory_samples = simulate(
        t_grid=t_grid,
        omega_distr=omega_distr,
        n_samples=10,
        model_kwargs=model_kwargs,
        init_cond=init_cond,
        rule="random",
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for sample_solution in trajectory_samples:
        ax.plot(t_grid, sample_solution, alpha=0.8)

    ax.plot(
        t_grid,
        deterministic_solution,
        linestyle="--",
        linewidth=2,
        label="Deterministic solution",
    )


    ax.set_xlabel("t")
    ax.set_ylabel("y(t)")
    ax.set_title("Ten oscillator trajectories for random omega samples")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig("assignment_4_trajectories.png", bbox_inches="tight")

    plt.show()
    # ====================================================================