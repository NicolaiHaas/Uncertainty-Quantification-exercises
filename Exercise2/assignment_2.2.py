import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

mc_samples = 1000000

'''
    Generates Hermite or Legendre polynomias up to and including degree n. Then returns the expected product
    of any two polynomials p_i(x) * p_j(x) w.r.t. x from the given distribution. The global mc_samples
    regulates the precision of the approximation on the expected value.
    Args:
        distr: cp.Distribution          - The distribution of x / indicator for the 
                                          type of polynomial to generate. Can be 
                                          cp.Normal or cp.Uniform
        n: int                          - The maximum polynomial dergree
    
    Returns:
        expected_prod: np.array (n,n)   - The matrix of expected product between every polynomial
'''
def calculate_polynoms_product(distr: cp.Distribution, n: int) -> npt.NDArray:
    # TODO: generate the first n orthonormal polynomials w.r.t. the given
    # distribution and compute the expected value of their inner products.
    # ====================================================================
    is_normal = isinstance(distr,cp.Normal)

    assert is_normal or isinstance(distr,cp.Uniform), \
        "Can only generate Legendre/Hermite for Uniform/Normal distributions"

    if is_normal:
        mode = 'probabilist'
        P = [np.array([np.pi**-0.25]),np.array([0,np.sqrt(2)*np.pi**-0.25])] if mode == 'physicist' else [np.array([1]),np.array([0,1])]
        
        if n < 2:
            P = P if n == 1 else P[0]
    
        elif mode == 'physicist':
            for i in range(1,n):
                P.append(
                    (np.sqrt(2)*np.insert(P[i],0,0) - np.sqrt(i)*np.append(P[i-1],[0,0]))/np.sqrt(i+1)
                )
        elif mode == "probabilist":
            for i in range(1,n):
                P.append(
                    (np.insert(P[i],0,0) - np.sqrt(i)*np.append(P[i-1],[0,0]))/np.sqrt(1+i)
                )
    # Case uniform -> Generate Legendre
    else:
        P = [np.array([1]), np.array([0, np.sqrt(3)])] # First two orthnormal Legednre Polynomials as list of coefficients 
        if n < 2:
            P = P if n == 1 else P[0]
        
        # Bonnet's recursion P_i+1 = ((2i+1)xP_i - iP_i-1)/i+1 adjusted for orthonormal polys
        # Multiplication with x is equivalent to insertion of zero coefficient in front / shift coefficients right
        else:
            for i in range(1,n):
                P.append(
                    np.sqrt(2*i+3)*(np.sqrt(2*i+1)/(i+1)*np.insert(P[i],0,0) - i/((i+1)*np.sqrt(2*i-1))*np.append(P[i-1],[0,0]))
                )

    # Compute O_ij = E_{x ~ d}[P_i(x)*P_j(x)] = dirac_ij through Monte Carlo approximation from distribution d
    samples = distr.sample(mc_samples)
    
    # Rescale samples s.t. it fits the orthonormal polys (same as scaling x in p(x), the polynomial)
    prm_dict = distr._parameters
    if is_normal:
        mu = prm_dict["shift"]
        sigma = prm_dict["scale"]
        samples = (samples - mu)/sigma
    else:
        a = prm_dict["lower"]
        b = prm_dict["upper"]
        samples =  2*(samples-a)/(b-a) -1
    # Repeat and cumprod to get polynomial features. Then vstack with 1-vec for constant term. Features are then (samples,n+1)
    feature_vec = np.repeat(samples[:,None], n,axis=1).cumprod(axis=1)
    feature_vec = np.hstack([np.ones((mc_samples,1)),feature_vec])

    # Polynomial evaluation x@coefficients for polynomials i and j. Then average for MC estimate. 
    expected_prod = np.asarray([[np.mean((feature_vec[:,:i+1]@P[i]) * (feature_vec[:,:j+1]@P[j])) for i in range(len(P))] for j in range(len(P))])
    
    # ====================================================================
    return expected_prod


if __name__ == "__main__":
    # Define the parameters of the simulation
    # ====================================================================
    
    np.random.seed(42)
    rho1 = cp.Uniform(-1,1)
    rho2 = cp.Normal(5,1)
    N = 10

    # ====================================================================
    # Compute the inner products.
    # ====================================================================

    O_uniform = calculate_polynoms_product(rho1,N)
    O_normal = calculate_polynoms_product(rho2,N)

    # ====================================================================
    # Visualize the results.
    # ====================================================================
    
    # You could look at the matrices itself
    fig, axs = plt.subplots(1,2)
    mp = plt.cm.ScalarMappable(cmap="Greys")
    im_uni =axs[0].imshow(O_uniform,cmap = "Greys")
    im_norm = axs[1].imshow(O_normal,cmap="Greys")

    axs[0].set_title("Product of orthonormal Legendre Polynomials")
    axs[1].set_title("Product of orthonormal Hermite Polynomials")
    axs[0].set_xlabel("Degree")
    axs[1].set_xlabel("Degree")
    axs[0].set_ylabel("Degree")
    axs[1].set_ylabel("Degree")
    fig.colorbar(im_norm, ax=axs)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

    # Or on the norm of O as the MC approximation gets more samples:
    uniform_ls = []
    normal_ls = []
    n_ls = [1000,10000,100000,1000000]
    for n in n_ls :
        mc_samples = n
        uniform_ls.append(
            np.linalg.norm(calculate_polynoms_product(rho1,N)-np.eye(N+1),ord="fro")
            )
        normal_ls.append(
            np.linalg.norm(calculate_polynoms_product(rho2,N)-np.eye(N+1),ord="fro")
            )
    uni_map = plt.plot(n_ls,uniform_ls)[0]
    plt.xlabel("Number of MC smaples")
    plt.ylabel(r"$\|\mathbb{E}_{x \sim \rho}[\phi_i(x) \phi_j(x)] - \mathbb{1}\|_{Fro}$")
    plt.title("Difference between polynomial inner product matrix $\Phi_{ij}$ and Identity")
    plt.xscale('log')
    plt.yscale('log')
    norm_map = plt.plot(n_ls,normal_ls)[0]
    plt.legend([uni_map,norm_map],[r"$\rho = \mathcal{U}(-1,1)$", r"$\rho = \mathcal{N}(5,1)$"])
    plt.show()
    # ====================================================================
