import numpy as np
import numpy.typing as npt
import scipy.special as sp

from utils.oscillator import Oscillator

'''
    Generates a grid of nodes for interpolation
    Args:
        bounds :    (float,float)  - defines interval on which nodes are spread
        n_nodes:    int            - number of nodes on the grid
        grid_type:  str            - type of grid is either "uniform" or "chebyshev"

    Returns:
        grid:       np.array       - the ordered grid of interpolation nodes.
'''
def generate_grid(
    bounds: tuple[float, float], n_nodes: int, grid_type: str = "uniform"
) -> npt.NDArray:
    # ====================================================================
    a,b = bounds
    N = n_nodes
    grid = np.linspace(a,b,N)
    assert grid_type in ["uniform", "chebyshev"], "Grid type not valid"
    match grid_type:
        case 'uniform':
            grid = grid
        case 'chebyshev':
            grid = ((np.cos((np.arange(N)+0.5)*np.pi/N) +1) * (b-a)/2 + a)[::-1]
    # ====================================================================
    return grid

'''
    Calculates the relative error of the empirical mean and variance of the samples w.r.t. the given
    reference values.
    Args:
        samples: np.array       - the sample data for which the QoI's are calculated
        mean_ref: float         - reference mean
        var_ref: float          - reference variance

    Returns:
        mean_error: float       - relative error of empirical mean
        var_error : float       - relative error of empirical variance 
'''
def compute_errors(
    samples: npt.NDArray, mean_ref: float, var_ref: float
) -> tuple[float, float]:
    # ====================================================================
    emp_mean = samples.mean()
    emp_var = samples.var(mean=emp_mean,ddof=1)
    
    mean_error = np.abs((mean_ref-emp_mean)/mean_ref)
    var_error = np.abs((var_ref-emp_var)/var_ref)

    # ====================================================================
    return mean_error, var_error

'''
    Loads two values from the file specified by the argument.
    Args:
        filename: str       - The relative path to the data file

    Returns:
        mean: float         - The mean in the given file (first line)
        var: float          - The variance in the given file (second line)
'''
def load_reference(filename: str) -> tuple[float, float]:
    # ====================================================================
    with open(filename,'r') as fd:
        mean = float(fd.readline())
        var = float(fd.readline())
    # ====================================================================
    return mean, var

'''
    Creates solutions for the ODE of an oscillator with given coefficients and 
    an array (1D) of frequencies, each of which recieves solutions for the nodes in t_grid.
    Args:
        t_grid: np.array                 - The nodes for which ODE solutions are calculated.
        omega_samples: np.array          - The list of frequencies for which solutions are created.
        model_kwargs: dict[str -> float] - Contains all the equations parameters except the mode of solving 
                                           and initial conditions see [oscillator.py] for details.
        init_cond: dict[str -> float]    - Contains the initial conditions.
        progress: bool                   - Prints progress to stdout.
    Returns:
        sample_solutions: np.array       - Array (|omegas| x |t_grid|) that contains solutions for each omega.
'''
def simulate(
    t_grid: npt.NDArray,
    omega_samples: npt.NDArray,
    model_kwargs: dict[str, float],
    init_cond: dict[str, float],
    progress: bool = False
) -> npt.NDArray:
    # ====================================================================
    assert omega_samples.ndim == 1 , "Only one-dimesional list of frequencies permitted"
    assert t_grid.ndim == 1, "Only one-dimensional time grid allowed for finding ODE solutions"
    
    c = model_kwargs.get('c',1.0)
    k = model_kwargs.get('k',1.0)
    f = model_kwargs.get('f',1.0)

    atol= model_kwargs.get('atol',1e-10)
    rtol= model_kwargs.get('rtol',1e-10)

    y0 = init_cond.get('y0',1.0)
    y1 = init_cond.get('y1',1.0)

    # list of discretized oscillators for every frequency omega
    t_grid_ls = []
    if progress: print("Starting simulation:\n Progress:")
    for i,omega in enumerate(omega_samples):
        if progress and i % 1000 == 0: print(f"  {i/len(omega_samples)*100:.2f}%",end="\r")
        osci = Oscillator(c,k,f,omega)
        t_grid_ls.append(osci.discretize("odeint",y0,y1,t_grid,atol,rtol))
    if progress: print("Simulation done!")
    sample_solutions = np.asarray(t_grid_ls)
    # ====================================================================
    return sample_solutions
