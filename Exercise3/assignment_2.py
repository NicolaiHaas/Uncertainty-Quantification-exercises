from functools import partial

from matplotlib.colors import Normalize

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing import Callable


def exp_cov_fn(x: npt.NDArray, y: npt.NDArray, scale: float) -> npt.NDArray:
    """Computes the exponential covariance function between two sets of points."""

    # (N^2,2) each
    flat_x = x.reshape(-1,2)
    flat_y = y.reshape(-1,2)

    pair_distances = flat_x[:,None] - flat_y[None,:] 
    dist = np.linalg.norm((pair_distances),ord=2,axis=-1)/scale
    return np.exp(-dist)


def squared_exp_cov_fn(x: npt.NDArray, y: npt.NDArray, scale: float):
    """Computes the squared exponential covariance function between two sets of points."""
    
    # (N^2,2) each
    flat_x = x.reshape(-1,2)
    flat_y = y.reshape(-1,2)

    pair_squared_distances = (flat_x[:,None] - flat_y[None,:])**2 
    dist = np.sum(pair_squared_distances, axis = -1)/(2*scale**2)
    return np.exp(-dist)


def get_xy_mesh(
    x_lims: tuple[float, float],
    y_lims: tuple[float, float],
    x_mesh_size: int,
    y_mesh_size: int,
) -> npt.NDArray:
    """Creates a 2D mesh grid for the given limits and mesh sizes."""
    x_step = (x_lims[1] - x_lims[0]) / x_mesh_size
    y_step = (y_lims[1] - y_lims[0]) / y_mesh_size
    x_grid = np.arange(x_lims[0] + x_step / 2, x_lims[1], x_step)
    y_grid = np.arange(y_lims[0] + y_step / 2, y_lims[1], y_step)
    mesh = np.stack(np.meshgrid(x_grid, y_grid), axis=-1)
    return mesh


def sample(mesh : np.ndarray,
    mean_fn : Callable, 
    cov_fn : Callable, 
    n_samples : int, 
    rng : np.random.Generator, 
    reg_scale=1e-7
) -> np.ndarray:            
    """Samples from a Gaussian process defined by the mean and covariance functions."""
    N = mesh.shape[0]
    mean = mean_fn(mesh) # N^22 vector of 0.1
    cov = cov_fn(mesh,mesh) # (N^2,N^2) matrix
    cholesky_cov = np.linalg.cholesky(cov+np.eye(len(cov))*reg_scale)

    # n samples with size N^2 each
    normal_samples = rng.multivariate_normal(np.zeros(N**2), np.eye(N**2),n_samples)
    
    # (N^2,1) + (N^2,N^2)@(N^2,n) = (N^2,n)
    G_i = mean[:,None] + cholesky_cov@normal_samples.T
    
    # shape (n,N,N)
    G_i = G_i.reshape(N,N,n_samples).T
    return G_i


def plot_samples(samples, x_lims, y_lims, same_norm = True):
    """Plots the samples from the Gaussian process."""
    norm = Normalize(vmin=samples.min(), vmax=samples.max()) if same_norm else None
    n_plots = len(samples)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    for ax, sample in zip(axes, samples):
        ax.imshow(sample, cmap="coolwarm", origin="lower", extent=(*x_lims, *y_lims), norm=norm)
    return fig
    


if __name__ == "__main__":

    x_lims, y_lims = (0, 1), (0, 1)
    x_mesh_size, y_mesh_size =150, 150
    scale = 2
    mean = lambda x: np.zeros(x.shape[0]**2)+0.1
    seed = 42
    n_samples = 3
    rng = np.random.default_rng(seed)

    # shape (100,100,2)
    mesh = get_xy_mesh(x_lims,y_lims,x_mesh_size,y_mesh_size)

    cov_fn = lambda x,y: exp_cov_fn(x,y,scale)
    exp_samples = sample(mesh,mean,cov_fn,n_samples,rng)

    cov_fn = lambda x,y: squared_exp_cov_fn(x,y,scale)
    squared_exp_samples = sample(mesh,mean,cov_fn,n_samples,rng)

    exp_fig = plot_samples(exp_samples,x_lims,y_lims)
    squared_exp_fig = plot_samples(squared_exp_samples,x_lims,y_lims, same_norm=False)



    exp_fig.colorbar(exp_fig.axes[0].images[0],ax= exp_fig.axes, label="Value")
    for i,ax in enumerate(squared_exp_fig.axes):
        squared_exp_fig.colorbar(ax.images[0],ax=ax, label="Value")

    exp_fig.suptitle("Exponential Kernel")
    squared_exp_fig.suptitle("Squared Exponential Kernel")

    plt.show()