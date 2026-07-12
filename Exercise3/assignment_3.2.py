import time
from functools import partial
from typing import Callable

import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.oscillator import Oscillator
from utils.wiener import WienerProcess


def generate_f_samples(
    mu: float,
    t_grid: npt.NDArray,
    n_samples: int,
    M: int | None,
    rng: np.random.Generator,
) -> list[Callable[[float], float]]:
    """Generates samples of the Wiener process."""

    # TODO: generate realizations of the Wiener process for f(t).
    # If M is None, we generate samples using the standard definition.
    # If M is not None, we generate samples using the KL expansion with M terms.
    # The samples are returned as a list of callable functions that
    # evaluate the Wiener process at a given time point.

    wiener = WienerProcess(mu, t_grid=t_grid)

    samples = []
    if M is not None:
        samples = wiener.approximate_kl(n_samples, M, rng)
    else:
        samples = wiener.generate(n_samples, rng)

    # see if slicing works
    dt = t_grid[1] - t_grid[0]

    def gen_fun(sample):
        def eval(t):
            # print(t)
            index = int(np.floor(t / dt))
            # print(index)
            return sample[index]
        return eval
    
    # return [lambda t, sample=sample: sample[int(np.floor(t / dt)) - 1] for sample in samples]
    return [gen_fun(sample) for sample in samples]
    

def simulate(
    t_grid: npt.NDArray,
    f_samples: list[Callable[[float], float]],
    model_kwargs: dict[str, float],
    init_cond: dict[str, float],
) -> npt.NDArray:
    """Simulates the oscillator model for each sample of f(t)."""

    # TODO: simulate the oscillator model for each sample of f(t) and
    # return the trajectories as 2D array.

    # do for each sample
    # odeint is so slow
    path_solves = np.zeros((len(f_samples), len(t_grid)))
    for i, sample in enumerate(f_samples):
        osci = Oscillator(model_kwargs["c"], model_kwargs["k"], sample, model_kwargs["omega"])

        path_solves[i, :] = osci.discretize("euler", init_cond["y0"], init_cond["y1"], t_grid)

    return path_solves


def compute_metrics(solutions: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Computes the mean and standard deviation of the solutions."""

    # TODO: compute the metrics.
    # ?
    # this needs to handle 2d arrays?
    mean = np.mean(solutions, axis=0)
    standard = np.std(solutions, ddof=1, axis=0)
    return (mean, standard)


def plot_solutions(
    t_grid: npt.NDArray, sampler_solutions: dict[str, npt.NDArray]
) -> plt.Figure:
    """Plots the oscillator trajectories for each sample of f."""
    n_plots = len(sampler_solutions)
    fig, axes = plt.subplots(
        1, n_plots, figsize=(6 * n_plots, 4), sharex=True, sharey=True
    )
    for ax, (name, solutions) in zip(axes, sampler_solutions.items()):
        mean, std = compute_metrics(solutions)
        ax.plot(t_grid, solutions.T, alpha=0.01, c="b")
        ax.plot(t_grid, mean, c="r", label="mean")
        ax.fill_between(
            t_grid, mean - std, mean + std, color="red", alpha=0.5, label="std"
        )

        # Add legend for samples manually.
        handles, _ = ax.get_legend_handles_labels()
        line = lines.Line2D([0], [0], color="b", label="Monte Carlo samples")
        handles.append(line)
        ax.legend(handles=handles)

        ax.set_title(name)
    return fig


if __name__ == "__main__":
    # TODO: set parameters of the model.
    f_mean = 0.5
    model_kwargs = {"c": 0.5, "k": 2.0, "omega": 1.0}
    init_cond = {"y0": 0.5, "y1": 0}

    # TODO: set the time domain.
    T_max = 10
    dt = 0.01
    t_grid = np.arange(0, T_max + dt, dt)

    # TODO: set the number of Monte-Carlo samples and KL terms.
    N = 1000
    Ms = [5, 10, 100]
    seed = 1234
    rng = np.random.default_rng(seed)

    ###########################################################################

    # TODO: generate samples of the Wiener process for f using the stadard
    # generation and the KL expansion for different M.
    
    # list of functions
    sam_evals_wiener = generate_f_samples(f_mean, t_grid, M= None, n_samples=N, rng=rng)
    # a list of lists of functions
    # sam_evals_list_kl = [generate_f_samples(f_mean, t_grid, M=M, n_samples=N, rng=rng) for M in Ms]
    # do indiv to reset rng
    rng = np.random.default_rng(seed)
    sam_evals_5_kl = generate_f_samples(f_mean, t_grid, M=5, n_samples=N, rng=rng)
    rng = np.random.default_rng(seed)
    sam_evals_10_kl = generate_f_samples(f_mean, t_grid, M=10, n_samples=N, rng=rng)
    rng = np.random.default_rng(seed)
    sam_evals_100_kl = generate_f_samples(f_mean, t_grid, M=100, n_samples=N, rng=rng)


    # TODO: simulate the oscillator model for each sample of f and record the
    # mean and standard deviation of the solutions at T_max.
    wien_paths = simulate(t_grid, sam_evals_wiener, model_kwargs, init_cond)

    kl_5_paths = simulate(t_grid, sam_evals_5_kl, model_kwargs, init_cond)
    kl_10_paths = simulate(t_grid, sam_evals_10_kl, model_kwargs, init_cond)
    kl_100_paths = simulate(t_grid, sam_evals_100_kl, model_kwargs, init_cond)


    mean_wien, var_wien = compute_metrics(wien_paths[:, -1])
    # print(wien_paths[:, -1].shape)
    mean_kl5, var_kl5 = compute_metrics(kl_5_paths[:, -1])
    mean_kl10, var_kl10 = compute_metrics(kl_10_paths[:, -1])
    mean_kl100, var_kl100 = compute_metrics(kl_100_paths[:, -1])

    # print(f"{mean_wien};{var_wien}")
    # print(f"{mean_kl5};{var_kl5}")
    # print(f"{mean_kl10};{var_kl10}")
    # print(f"{mean_kl100};{var_kl100}")

    # TODO: optionally, plot the solutions for each sample of f.

    sampler_solutions = {"wien":wien_paths, "kl5":kl_5_paths, "kl10":kl_10_paths, "kl100":kl_100_paths}

    comp_fg = plot_solutions(t_grid, sampler_solutions)
    plt.show()

    print("wien", np.var(wien_paths[:, -1], ddof=1))
    print("5", np.var(kl_5_paths[:, -1], ddof=1))
    print("10", np.var(kl_10_paths[:, -1], ddof=1))
    print("100", np.var(kl_100_paths[:, -1], ddof=1))


    for name, solutions in sampler_solutions.items():
        _, std = compute_metrics(solutions)
        plt.plot(t_grid, std, label=name)

    plt.xlabel("t")
    plt.ylabel("std(y)")
    plt.legend()
    plt.show()
