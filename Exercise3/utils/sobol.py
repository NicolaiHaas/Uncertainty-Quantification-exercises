import chaospy as cp
import numpy as np
import numpy.typing as npt

from .oscillator import Oscillator


def _evaluate_oscillator(
    samples: npt.NDArray, t_grid: npt.NDArray, fixed_args: dict[str, float]
) -> npt.NDArray:
    """Evaluates the oscillator model for given samples."""
    c = fixed_args['c']
    k = fixed_args['k']
    f = fixed_args['f']
    y0 = fixed_args['y0']
    y1 = fixed_args['y1']
    t_grid = fixed_args['t_grid']
    atol = fixed_args.get('atol',1e-10)
    rtol = fixed_args.get('rtol',1e-10)

    solutions = []
    for omega in samples:
        solutions.append(
            Oscillator(c,k,f,omega).discretize('odeint',y0,y1,t_grid,atol,rtol)
            )
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
    N = n_samples

    # N x K sample matrices 
    A = distribution.sample(N//2, rule="sobol")
    B = distribution.sample(N//2, rule="sobol")

    # N x K as t_grid is collapsed to size one
    f_A = _evaluate_oscillator(A,t_grid,fixed_args)[-1]
    f_B = _evaluate_oscillator(B,t_grid,fixed_args)[-1]
    f_AB = np.zeros([K,N,K])
    S_T = np.zeros([K])
    S = np.zeros([K])

    # Variance of the output. No specification in the paper so that is
    # the best I can do
    var_Y = np.var(np.stack([f_A,f_B]),ddof=1)

    for i in range(K):
        AB_i = A.copy()
        AB_i[i] = B[i]
        f_AB[i] = _evaluate_oscillator(AB_i,t_grid,fixed_args)[-1]
        S_T[i] = np.sum((f_A - f_AB[i])**2, axis=0)/(2*N)           # Eq. 19
        S[i] = np.sum(f_B * (f_AB[i] - f_A), axis=0)/N              # adjusted Eq. 16

    return S/var_Y, S_T/var_Y


def pseudo_spectral_sobol(
    pce_degree: int,
    quadrature_degree: int,
    distribution: cp.Distribution,
    t_grid: npt.NDArray[np.float64],
    fixed_args: dict[str, float],
    sparse=True,
) -> tuple[float, float]:
    """Computes the Sobol' indices using a pseudo-spectral method."""
    
    # TODO: implement the pseduo-spectral method.
    return 0, 0
