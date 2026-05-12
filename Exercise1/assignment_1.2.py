import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.sampling import compute_rmse


def sample_normal(
    n_samples: int, mu_target: npt.NDArray, V_target: npt.NDArray, seed: int = 42
) -> npt.NDArray:
    # TODO: generate samples from multivariate normal distribution.
    # ====================================================================
    rng = np.random.default_rng(seed)

    # NumPy returns samples with shape (N, d). We transpose the result
    # to obtain the convention (d, N) used throughout the worksheet.
    samples = rng.multivariate_normal(
        mean=mu_target,
        cov=V_target,
        size=n_samples,
    ).T
    # ====================================================================
    return samples


def compute_moments(samples: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    # TODO: estimate mean and covariance of the samples.
    # ====================================================================

    mean = np.mean(samples,axis=1)
    covariance = np.cov(samples, ddof=1)

    # ====================================================================
    return mean, covariance


if __name__ == "__main__":
    # TODO: define the parameters of the simulation.
    # ====================================================================
    mu_target = np.array([-0.4, 1.1])

    var_target = np.array(
        [
            [2.0, 0.4],
            [0.4, 1.0],
        ]
    )

    sample_sizes = [10, 100, 1000, 10000]
    # ====================================================================

    # TODO: estimate mean, covariance, and compute the required errors.
    # ====================================================================
    mean_errors = []
    diagonal_cov_errors = []
    off_diagonal_cov_errors = []
    mean_rmses = []

    for n_samples in sample_sizes:
        samples = sample_normal(
            n_samples=n_samples,
            mu_target=mu_target,
            V_target=var_target,
        )

        mean, covariance = compute_moments(samples)

        mean_errors.append(abs(mean[0] - mu_target[0]))

        diagonal_cov_errors.append(
            abs(covariance[0, 0] - var_target[0, 0])
        )

        off_diagonal_cov_errors.append(
            abs(covariance[0, 1] - var_target[0, 1])
        )

        # RMSE according to equation (1) from the worksheet
        mean_rmses.append(compute_rmse(samples)[0])

        print(f"\nN = {n_samples}")
        print(f"Estimated mean:\n{mean}")
        print(f"Estimated covariance:\n{covariance}")

    mean_errors = np.array(mean_errors)
    diagonal_cov_errors = np.array(diagonal_cov_errors)
    off_diagonal_cov_errors = np.array(off_diagonal_cov_errors)
    mean_rmses = np.array(mean_rmses)
    # ====================================================================

    # TODO: plot the results on the log-log scale.
    # ====================================================================
    plt.figure(figsize=(7, 5))

    plt.plot(sample_sizes, mean_errors, marker="o", label="Mean absolute error")
    plt.plot(sample_sizes, diagonal_cov_errors, marker="o", label="Diagonal covariance error")
    plt.plot(sample_sizes, off_diagonal_cov_errors, marker="o", label="Off-diagonal covariance error")
    plt.plot(sample_sizes, mean_rmses, marker="o", linestyle="--", label="Mean RMSE")

    plt.xlabel("Number of samples N")
    plt.ylabel("Error")
    plt.title("Monte Carlo sampling to estimate mean and covariance")
    plt.grid(True, which="both")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/assignment_1.2.png", bbox_inches="tight")
    plt.show()
    # ====================================================================
