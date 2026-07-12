import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.wiener import WienerProcess


def plot_eigenpairs(
    wiener: WienerProcess, n_terms: int, t_grid: npt.NDArray[np.float64]
) -> plt.Figure:
    """Plots the first n_terms eigenvalues and eigenfunctions of the Wiener process."""
    eigenvalues, eigenfunctions = wiener.kl_eigenpairs(n_terms)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(np.arange(1, n_terms + 1), eigenvalues, marker="o")
    axes[0].set_yscale("log")
    axes[0].set_title(f"First {n_terms} eigenvalues")
    axes[1].plot(t_grid, eigenfunctions(t_grid))
    axes[1].set_title(f"First {n_terms} eigenfunctions")
    return fig


if __name__ == "__main__":
    # TODO: set the configuration.
    T = 1.0
    n_points = 1000
    t_grid = np.linspace(0, T, n_points)
    Ms = [10, 100, 1000]
    seed = 1234
    n_samples = 100
    rng = np.random.default_rng(seed)

    org_wiener = WienerProcess(mu=0.0, T = T, n_points=n_points)

    # TODO: generate one realization of the Wiener process using the
    # standard definition.
    rng = np.random.default_rng(seed)
    single_standard = org_wiener.generate(1, rng)[0]

    # plot direct path
    plt.plot(t_grid, single_standard)
    plt.xlabel("t")
    plt.ylabel(r"$W$")
    plt.legend()
    plt.show()

    # TODO: generate approximations of the Wiener process using the KL expansion.
    # generate ev with M = 1000
    M = Ms[-1]
    lamdas = org_wiener.kl_eigenvalues(M)
    
    # simpler plot for just the evs, but all 1000
    plt.plot(np.asarray(range(M)) + 1, lamdas)
    plt.yscale("log")
    plt.xlabel("M")
    plt.ylabel(r"$\lambda_m$")
    plt.show()



    # kl_paths = org_wiener.approximate_kl(n_samples, Ms[-1], rng)

    # TODO: plot the approximation results.
    
    # generate for all Ms
    # collect
    kl_paths = {}

    for M in Ms:
        # reseed
        rng = np.random.default_rng(seed)
        # need to remove empty col
        kl_paths[M] = org_wiener.approximate_kl(1, M, rng,)[0]

    # plot paths ove M
    fig, axes = plt.subplots(1, 3, figsize = (12, 3))
    for M, ax in zip(kl_paths, axes):
        ax.plot(t_grid, kl_paths[M], label=f"M = {M}")
        ax.set_xlabel("t")
        ax.legend()
    axes[0].set_ylabel(r"$W$")
    fig.show()


    # plot org vs M1000
    plt.plot(t_grid, single_standard, label="Wiener")
    plt.plot(t_grid, kl_paths[1000], label="KL(1000)")
    plt.xlabel("t")
    plt.ylabel(r"$W$")
    plt.legend()
    plt.show()
    # with different seed
    rng = np.random.default_rng(seed)
    single_wiener_ns = org_wiener.generate(3, rng)
    rng = np.random.default_rng(seed)
    single_kl_ns = org_wiener.approximate_kl(3, M, rng,)

    for i in range(3):
        plt.plot(t_grid, single_wiener_ns[i], color = "green",
                label="Wiener" if i == 0 else None)
        plt.plot(t_grid, single_kl_ns[i], color = "purple",
                label="KL(1000)" if i == 0 else None)
    plt.xlabel("t")
    plt.ylabel(r"$W$")
    plt.legend()
    plt.show()


    # TODO: visualize first eigenvalues and eigenfunctions.
    # show first 5 for functions
    ep_plt = plot_eigenpairs(org_wiener, 5, t_grid)
    plt.show()
    