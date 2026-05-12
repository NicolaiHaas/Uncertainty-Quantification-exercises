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
    return np.e - 1.0
    # ====================================================================


def run_monte_carlo(Ns: tuple[int, ...], seed: int = 42) -> list[float]:
    # TODO: run the Monte Carlo method and return the absolute error
    # of the estimation.
    # ====================================================================
    distribution = cp.Uniform(0.0, 1.0)
    exact_value = analytical_integral()

    errors = []

    for n_samples in Ns:
        estimate, _ = monte_carlo(p=distribution, n_samples=n_samples,
            f=f, seed=seed)

        errors.append(abs(estimate[0] - exact_value))

    return errors
    # ====================================================================


def run_control_variates(Ns: tuple[int, ...], seed: int = 42):
    # TODO: run the control variate method for and return the absolute
    # errors of the resulting estimations.
    # ====================================================================
    distribution = cp.Uniform(0.0, 1.0)
    exact_value = analytical_integral()

    controls = [
        (lambda x: x, 0.5),
        (lambda x: 1.0 + x, 1.5),
        (lambda x: 1.0 + x + 0.5 * x**2, 1.0 + 0.5 + 1.0 / 6.0),
    ]

    all_errors = []

    for phi, control_mean in controls:
        errors = []

        for n_samples in Ns:
            estimate = control_variates(p=distribution, n_samples=n_samples,
                f=f, phi=phi, control_mean=control_mean, seed=seed)

            errors.append(abs(estimate[0] - exact_value))

        all_errors.append(errors)

    return tuple(all_errors)
    # ====================================================================


def run_importance_sampling(
    Ns: tuple[int, ...], seed: int = 42
):
    # TODO: run the importance sampling method and return the absolute
    # errors of the resulting estimations.
    # ====================================================================
    target_distribution = cp.Uniform(0.0, 1.0)
    proposal_distributions = [
        cp.Beta(5.0, 1.0),
        cp.Beta(0.5, 0.5),
    ]

    exact_value = analytical_integral()
    all_errors = []

    for proposal_distribution in proposal_distributions:
        errors = []

        for n_samples in Ns:
            estimate = importance_sampling(p=target_distribution,
                q=proposal_distribution, n_samples=n_samples,
                f=f, seed=seed
            )

            errors.append(abs(estimate[0] - exact_value))

        all_errors.append(errors)

    return tuple(all_errors)
    # ====================================================================


if __name__ == "__main__":
    # TODO: define the parameters of the simulation.
    # ====================================================================
    sample_sizes = (10, 100, 1000, 10000)
    # ====================================================================

    # TODO: run all the methods.
    # ====================================================================
    mc_errors = run_monte_carlo(sample_sizes)

    cv_h1_errors, cv_h2_errors, cv_h3_errors = run_control_variates(sample_sizes)

    is_beta_5_1_errors, is_beta_half_half_errors = run_importance_sampling(sample_sizes)

    # trying print as table
    print("Absolute errors\n")
    print(f"{'N':>8} | {'MC':>12} | {'CV h1':>12} | {'CV h2':>12} | "
          f"{'CV h3':>12} | {'IS β(5,1)':>12} | {'IS β(.5,.5)':>14}")

    for i, n_samples in enumerate(sample_sizes):
        print(
            f"{n_samples:8d} | "
            f"{mc_errors[i]:12.6e} | "
            f"{cv_h1_errors[i]:12.6e} | "
            f"{cv_h2_errors[i]:12.6e} | "
            f"{cv_h3_errors[i]:12.6e} | "
            f"{is_beta_5_1_errors[i]:12.6e} | "
            f"{is_beta_half_half_errors[i]:14.6e}"
        )
    # ====================================================================

    # TODO: plot the results on the log-log scale.
    # ====================================================================
    plt.figure(figsize=(8, 5))

    plt.plot(sample_sizes, mc_errors, marker="o", label="Standard MC")
    plt.plot(sample_sizes, cv_h1_errors, marker="o", label="Control variate h1")
    plt.plot(sample_sizes, cv_h2_errors, marker="o", label="Control variate h2")
    plt.plot(sample_sizes, cv_h3_errors, marker="o", label="Control variate h3")
    plt.plot(sample_sizes, is_beta_5_1_errors, marker="o", label="IS Beta(5, 1)")
    plt.plot(sample_sizes, is_beta_half_half_errors, marker="o", label="IS Beta(0.5, 0.5)")

    # Reference Monte Carlo convergence rate O(N^{-1/2}).
    plt.plot(
        sample_sizes,
        1 / np.sqrt(np.array(sample_sizes)),
        linestyle=":",
        color="black",
        label=r"$\mathcal{O}(N^{-1/2})$",
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Number of samples N")
    plt.ylabel("Absolute error")
    plt.title("Assignment 3: Monte Carlo variance reduction")
    plt.grid(True, which="both")
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/assignment_3.png", bbox_inches="tight")
    plt.show()
    # ====================================================================