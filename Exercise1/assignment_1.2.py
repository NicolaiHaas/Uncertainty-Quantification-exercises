import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.sampling import compute_rmse

#returns shape nxd
def sample_normal(
    n_samples: int, mu_target: npt.NDArray, V_target: npt.NDArray, seed: int = 42
) -> npt.NDArray:
    # generate samples from multivariate normal distribution.
    # ====================================================================
    np.random.seed(seed=seed)
    samples = np.random.multivariate_normal(mean=mu_target,cov=V_target,size=n_samples)
    # ====================================================================
    return samples


def compute_moments(samples: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    # estimate mean and covariance of the samples.
    # ====================================================================
    #recieves DxN samples
    mean = np.mean(samples,axis=1)
    covariance = np.cov(samples, ddof=1)
    # ====================================================================
    return mean, covariance


if __name__ == "__main__":
    # TODO: define the parameters of the simulation.
    # ====================================================================
    m = np.array([-0.4,1.1])
    V = np.array(
        [[2,0.4],
         [0.4,1]]
        )
    N = [10,100,1000,10000]
    # ====================================================================

    # TODO: estimate mean, covariance, and compute the required errors.
    # ====================================================================
    m_prime_ls = []
    V_prime_ls = []
    mean_RMSE_ls = []
    for n in N:
        samples = sample_normal(n,mu_target=m,V_target=V)
        m_prime, V_prime = compute_moments(samples.T)
        MSE = np.var(samples-m,axis=0,ddof=1)
        m_prime_ls.append(m_prime)
        V_prime_ls.append(V_prime)
        mean_RMSE_ls.append(np.sqrt(MSE/n))
    m_prime_ls = np.array(m_prime_ls)
    V_prime_ls = np.array(V_prime_ls)
    m_differences = abs(m_prime_ls - m[np.newaxis,:])
    V_differences = np.abs(V_prime_ls - V[None,:,:])
    mean_RMSE_ls = np.array(mean_RMSE_ls) 
    # ====================================================================
    # TODO: plot the results on the log-log scale.
    # ====================================================================
    # plot for first mean and first element of covariance matrix
    first_mean_plot = plt.plot(N,m_differences[:,0], ls = 'dotted', c= 'green')[0]
    first_cov_plot = plt.plot(N,V_differences[:,0,0], ls = 'dotted', c = 'blue')[0]
    first_RMSE_plot = plt.plot(N,mean_RMSE_ls[:,0], ls = 'dotted', c = 'red')[0]
    
    #plot norm of every dimension's means and covariances
    mean_norm_plot = plt.plot(N,np.linalg.norm(m_differences,axis=1), c = 'green')[0]
    cov_norm_plot = plt.plot(N,np.linalg.norm(V_differences,ord='fro',axis=(1,2)), c = 'blue')[0] 
    RMSE_norm_plot = plt.plot(N,np.linalg.norm(mean_RMSE_ls,axis=1), c = 'red')[0]
    
    plt.legend([first_mean_plot,first_cov_plot,first_RMSE_plot, mean_norm_plot,cov_norm_plot,RMSE_norm_plot],
              [r"$\Delta \mu_0$",r"$\Delta \Sigma_{0,0}$",r"RMSE($\mu_0$)", r"$|\Delta \mu|_2$",
               r"$|\Delta \Sigma|_F$", r"|RMSE($\mu$)|"]
              )
    plt.xscale('log')
    plt.xlabel("Number of samples")
    plt.ylabel("Error")
    plt.yscale("log")
    plt.show()

    # ====================================================================
