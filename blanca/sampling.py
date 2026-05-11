from typing import Callable

import chaospy as cp
import numpy as np
import numpy.typing as npt

Function = Callable[[npt.NDArray], npt.NDArray]


def compute_rmse(values: npt.NDArray) -> npt.NDArray:
    return np.std(values, ddof=1, axis=1) / np.sqrt(values.shape[1])


def monte_carlo(
    p: cp.Distribution,
    n_samples: int,
    f: Function,
    transform: Function | None = None,
    rule: str = "random",
    seed: float = 42,
) -> tuple[npt.NDArray, npt.NDArray]:
    # TODO: implement the Monte Carlo method.
    # Return the mean approximation and the corresponding RMSE. Make sure
    # the function works for both 1-dimensional and n-dimensional
    # distributions (see include_axis_dim parameter of
    # cp.Distribution.sample).
    # ====================================================================
    samples = p.sample(
        size=n_samples,
        rule=rule,
        seed=seed,
        include_axis_dim=True,
    )

    if transform is not None:
        samples = transform(samples)

    values = np.asarray(f(samples), dtype=float)
    values = np.atleast_2d(values)

    mean = np.mean(values, axis=1)
    rmse = compute_rmse(values)
    # ====================================================================
    return mean, rmse


def control_variates(
    p: cp.Distribution,
    n_samples: int,
    f: Function,
    phi: Function,
    control_mean: float,
    seed: float = 42,
) -> npt.NDArray:
    # TODO: implement the control variates method that returns the mean
    # approximation. Make sure the function works for both 1-dimensional
    # and n-dimensional distributions.
    # ====================================================================
    samples = p.sample(
        size=n_samples,
        seed=seed,
        include_axis_dim=True,
    )

    values = np.asarray(f(samples), dtype=float)
    controls = np.asarray(phi(samples), dtype=float)

    values = np.atleast_2d(values)
    controls = np.atleast_2d(controls)

    value_mean = np.mean(values, axis=1)
    control_sample_mean = np.mean(controls, axis=1)

    centered_values = values - value_mean[:, np.newaxis]
    centered_controls = controls - control_sample_mean[:, np.newaxis]

    covariance = np.mean(centered_values * centered_controls, axis=1)
    control_variance = np.mean(centered_controls**2, axis=1)

    beta = covariance / control_variance

    mean = value_mean - beta * (control_sample_mean - control_mean)
    # ====================================================================
    return mean


def importance_sampling(
    p: cp.Distribution,
    q: cp.Distribution,
    n_samples: int,
    f: Function,
    seed: float = 42,
) -> npt.NDArray:
    # TODO: implement the importance sampling that returns the mean
    # approximation. Make sure the function works for both 1-dimensional
    # and n-dimensional distributions.
    # ====================================================================
    samples = q.sample(
        size=n_samples,
        seed=seed,
        include_axis_dim=True,
    )

    values = np.asarray(f(samples), dtype=float)
    values = np.atleast_2d(values)

    # Importance weights p(x) / q(x).
    weights = np.asarray(
        p.pdf(samples) / q.pdf(samples),
        dtype=float,
    )

    weights = np.squeeze(weights)

    mean = np.mean(
        values * weights[np.newaxis, :],
        axis=1,
    )
    # ====================================================================
    return mean