import chaospy as cp
import numpy as np
import numpy.typing as npt

from .oscillator import Oscillator

def _evaluate_oscillator(
    samples: npt.NDArray, t_grid: npt.NDArray, fixed_args: dict[str, float]
) -> npt.NDArray:
    """Evaluates the oscillator model for given samples."""
    expected_args = ['c','k','f','omega','y0','y1']

    i = 0
    args = []
    for arg in expected_args:
        if arg in fixed_args:
            # if fixed take it from the dict
            args.append(fixed_args[arg])
        else:
            # If not fixed must be next in line in samples
            args.append(samples[i])
            i += 1
    
    c = args[0]
    k = args[1]
    f = args[2]
    omega = args[3]
    y0 = args[4]
    y1 = args[5]

    atol = fixed_args.get('atol',1e-10)
    rtol = fixed_args.get('rtol',1e-10)

    solutions = [Oscillator(c,k,f,omega).discretize('odeint',y0,y1,t_grid,atol,rtol)]
        
    return np.array(solutions)


def monte_carlo_sobol(
    n_samples: int,
    distribution: cp.Distribution,
    t_grid: npt.NDArray[np.float64],
    fixed_args: dict[str, float],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
        Computes the Sobol' indices using Monte Carlo sampling.
        Uses Eq. 19 and Eq. 16 of the paper
        "Variance based sensitivity analysis of model output. Design and estimator
        for the total sensitivity index" by A. Saltelli et al.
        Variable names go along the notation of the paper
    """
    K = len(distribution)
    n = n_samples

    uniform_2K = cp.Iid(cp.Uniform(0, 1), 2 * K)
    # (2K, N) as described in the paper
    A_and_B_unit = uniform_2K.sample(n, rule="sobol")  # Shape: (2*K, N)
    
    # (K, N) each
    A_unit = A_and_B_unit[:K, :]  
    B_unit = A_and_B_unit[K:, :]

    A = distribution.inv(A_unit)
    B = distribution.inv(B_unit)

    f_A = [_evaluate_oscillator(A[:, j], t_grid, fixed_args)[:, -1] for j in range(n)]
    f_B = [_evaluate_oscillator(B[:, j], t_grid, fixed_args)[:, -1] for j in range(n)]
    f_A = np.asarray(f_A).flatten()
    f_B = np.asarray(f_B).flatten()

    f_AB = np.zeros([K, n])
    S_T = np.zeros(K)
    S = np.zeros(K)

    # Variance of the output using the joined A/B ensemble
    var_Y = np.var(np.hstack([f_A, f_B]), ddof=1)

    for i in range(K):
        AB_i = A.copy()
        AB_i[i] = B[i]
        f_AB[i] = np.asarray([_evaluate_oscillator(AB_i[:, j], t_grid, fixed_args)[:, -1] for j in range(n)]).flatten()
        S_T[i] = np.sum((f_A - f_AB[i]) ** 2) / (2 * n)  # Eq. 19
        S[i] = np.sum(f_B * (f_AB[i] - f_A)) / n        # Eq. 16

    return S / var_Y, S_T / var_Y


def pseudo_spectral_sobol(
    pce_degree: int,
    quadrature_degree: int,
    distribution: cp.Distribution,
    t_grid: npt.NDArray[np.float64],
    fixed_args: dict[str, float],
    sparse=True,
) -> tuple[float, float]:
    """Computes the Sobol' indices using a pseudo-spectral method."""

    D = len(distribution)

    polys, norms = cp.generate_expansion(pce_degree,distribution, retall = True)
    
    # D x N for N nodes and D dimensions in distribution
    nodes, weights = cp.generate_quadrature(quadrature_degree,distribution,rule="gaussian", sparse=sparse)
    # if sparse, the i-th node is node[:,i] so solves is N-list
    solves = [_evaluate_oscillator(nodes[:,i], t_grid,fixed_args)[:,-1] for i in range(len(nodes[0]))]
    # check shape
    fit_polys = cp.fit_quadrature(polys,nodes,weights,solves, norms=norms)
    coeffs = np.asarray(fit_polys.coefficients).flatten()
    norms = np.asarray(norms).flatten()

    # Compute the Sobols using orthogonal polynomial norms
    S = np.zeros(D)
    S_T = np.zeros(D)
    for i in range(D):
        singular_mask = (
            (fit_polys.exponents[:, i] > 0)
            & np.all(fit_polys.exponents[:, [j for j in range(D) if j != i]] == 0, axis=1)
        )
        total_mask = fit_polys.exponents[:, i] > 0

        S[i] = np.sum(coeffs[singular_mask] ** 2 * norms[singular_mask])
        S_T[i] = np.sum(coeffs[total_mask] ** 2 * norms[total_mask])

    f_0 = coeffs[np.all(fit_polys.exponents == 0, axis=1)][0]
    var_Y = np.sum(coeffs ** 2 * norms) - f_0 ** 2 * norms[0]
    return S / var_Y, S_T / var_Y
