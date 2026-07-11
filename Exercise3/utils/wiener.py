from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class WienerProcess:
    mu: float
    T: float | None = None
    n_points: float | None = None
    t_grid: npt.NDArray | None = None

    def __post_init__(self):
        if self.T is None and self.n_points is None:
            self.T = self.t_grid[-1]
            self.n_points = len(self.t_grid)
        if self.t_grid is None:
            self.t_grid = np.linspace(0, self.T, self.n_points)

    def generate(self, n_samples: int, rng: np.random.Generator):

        # TODO: generate n_samples realizations of the Wiener process
        # using the standard definition.
        
        # t_grid given
        # calc dt
        dt = self.t_grid[1] - self.t_grid[0]

        # out mat
        w_paths_mat = np.zeros((n_samples, self.n_points))

        for i in range(n_samples):
            # all random steps
            # swrt(dt) * N(0,1)
            # one less to start at 0
            steps = np.sqrt(dt) * rng.standard_normal(self.n_points - 1)
            w_paths_mat[i,1:] = np.cumsum(steps)

        return w_paths_mat

    def approximate_kl(self, n_samples: int, M: int, rng: np.random.Generator):
        
        # TODO: generate n_samples realizations of the Wiener process
        # using the Karhunen-Loève expansion with M terms.

        W_paths_kl = np.zeros((n_samples, self.n_points))

        # draw for random coeffs
        C_m = rng.standard_normal((n_samples, M))
        # do these outside loop instead
        # eigenvalues
        lambda_sqr = np.sqrt(self.kl_eigenvalues(M))
        # eigenfunctions
        phis_t = self.kl_eigenfunctions(M)
        phis = phis_t(self.t_grid)

        # for each sample
        for i in range(n_samples):
            # sum over sqrt(lambda) * phi * Crandom
            # sum over modes
            # also other way around sum over cols
            path =  np.sum(lambda_sqr * phis * C_m[i, :], axis=1)
            W_paths_kl[i,:] = path

        return W_paths_kl

    def kl_eigenvalues(self, M: int):

        # TODO: compute the first M eigenvalues of the Wiener process.
        # 1 / ((m-0.5)^2 * pi^2)
        # vec of length
        m = np.asarray(range(M)) + 1
        lam = 1 / ((m - 0.5)**2 * np.pi**2) 
        return lam

    def kl_eigenfunctions(self, M: int):

        # TODO: compute the first M eigenfunctions of the Wiener process.
        # It might be more conveniet to return a callable function that
        # returns evaluations of the first M eigenfunctions for the provided
        # time points.

        sq2 = np.sqrt(2.0)
        m = np.asarray(range(M)) + 1

        # sqrt(2) * sin(pi*t*(m-0.5))
        def phi(t):
            # change dimenstions of t and m to get mat
            # other way around
            # t as cols, m as rows
            return sq2 * np.sin(np.pi * t[:, np.newaxis] * (m - 0.5)[np.newaxis, :])
        
        return phi

    def kl_eigenpairs(self, M: int):
        eigenvalues = self.kl_eigenvalues(M)
        eigenfunctions = self.kl_eigenfunctions(M)
        return eigenvalues, eigenfunctions
