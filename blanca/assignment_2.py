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
        distribution = cp.Uniform(0.0, 1.0)

        samples = distribution.sample(
            size=n_samples,
            seed=seed,
        )

        samples = transform(samples, a, b)

    else:
        distribution = cp.Uniform(a, b)

        samples = distribution.sample(
            size=n_samples,
            seed=seed,
        )

    function_values = f(samples)

    # The integral equals the interval length multiplied by the
    # expectation with respect to the uniform distribution.
    integral = (b - a) * np.mean(function_values)

    # The RMSE of the Monte Carlo estimator scales identically.
    rmse = (
        (b - a)
        * np.std(function_values, ddof=1)
        / np.sqrt(n_samples)
    )
    # ====================================================================
    return integral, rmse


if __name__ == "__main__":
    # TODO: define the parameters of the simulation.
    # ====================================================================
    Ns = [10, 100, 1000, 10000]

    intervals = [
        (0.0, 1.0, False, "Assignment 2.1: direct sampling on [0, 1]"),
        (2.0, 4.0, False, "Assignment 2.2: direct sampling on [2, 4]"),
        (2.0, 4.0, True, "Assignment 2.2: transformed sampling on [2, 4]"),
    ]
    # ====================================================================

    # TODO: compute the integral and the errors.
    # ====================================================================
    results = {}

    for a, b, with_transform, label in intervals:
        exact_value = analytical_integral(a, b)

        estimates = []
        errors = []
        rmses = []

        for n_samples in Ns:
            estimate, rmse = integrate_mc(
                f=f,
                a=a,
                b=b,
                n_samples=n_samples,
                with_transform=with_transform,
            )

            estimates.append(estimate)
            errors.append(abs(exact_value - estimate))
            rmses.append(rmse)

        results[label] = {
            "estimates": np.array(estimates),
            "errors": np.array(errors),
            "rmses": np.array(rmses),
            "exact": exact_value,
        }

        print(f"\n{label}")
        print(f"Exact integral: {exact_value:.8f}")

        for n_samples, estimate, error, rmse in zip(
            Ns,
            estimates,
            errors,
            rmses,
        ):
            print(
                f"N = {n_samples:5d} | "
                f"estimate = {estimate:.8f} | "
                f"error = {error:.8f} | "
                f"RMSE = {rmse:.8f}"
            )
    # ====================================================================

    # TODO: plot the results on the log-log scale.
    # ====================================================================
    # Compute global limits so all plots use identical axes and are
    # directly comparable.
    all_errors = []
    all_rmses = []

    for values in results.values():
        all_errors.extend(values["errors"])
        all_rmses.extend(values["rmses"])

    global_min = min(min(all_errors), min(all_rmses))
    global_max = max(max(all_errors), max(all_rmses))

    for label, values in results.items():
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.loglog(
            Ns,
            values["errors"],
            marker="o",
            label="Absolute error",
        )

        ax.loglog(
            Ns,
            values["rmses"],
            marker="o",
            linestyle="--",
            label="RMSE",
        )

        ax.set_xlim(min(Ns), max(Ns))
        ax.set_ylim(global_min, global_max)

        ax.set_xlabel("Number of samples N")
        ax.set_ylabel("Error")

        ax.set_title(label)

        ax.grid(True, which="both")
        ax.legend()

        fig.tight_layout()

        filename = (
            label.lower()
            .replace(":", "")
            .replace("[", "")
            .replace("]", "")
            .replace(",", "")
            .replace(" ", "_")
        )

        fig.savefig(f"{filename}.png", bbox_inches="tight")

    plt.show()
    # ====================================================================