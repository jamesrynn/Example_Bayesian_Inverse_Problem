
###########################
###  ---  MODULES  ---  ###
###########################
import math
import numpy as np
import matplotlib.pyplot as plt

import pickle

from scipy.stats import uniform as Unif
from scipy.stats import norm as Norm
from scipy.stats import multivariate_normal as mvNorm

from util_fig import myfigure        # plotting routine from Steven


#############################
###  ---  FUNCTIONS  ---  ###
#############################

def uxt(x,t,lam):
    """
    Evaluation of the function
        u(x,t;lambda) = (3sin(pi*x) + sin(3*pi*x))exp(-lambda*pi^2*t),
    which is the solution to the forward problem PDE.

    Parameters:
    -----------
    x : array_like
        Spatial locations at which function is to be evaluated.
    t : array_like
        Time points at which function is to be evaluated.
    lam : float
        Value of the thermal conductivity parameter.

    Returns:
    --------
    uxt : ndarray
        Value of the function u at each point in x and time in t for the given
        value of lambda.
    """

    # Spatial locations.
    X = np.array(3*np.sin(np.pi*x) + np.sin(3*np.pi*x))

    # Time points.
    T = np.array(np.exp(-lam * math.pow(np.pi,2) * t))

    # Compute solution through outer product and return values.
    return np.outer(T, X)
################################################################################



def m_s_from_mu_sig(mu,sig):
    """
    Convert hyperparameters (mu,sigma) of Gaussian distribution to (m,s)
    defining the logNormal distrubtion such that
        Y = exp(X): X~N(m,s^2); Y~logN(mu,sigma^2).

    Parameters:
    -----------
    mu : float
        Mean of the Gaussian distribution.
    sigma : float
        Standard deviation of the Gaussian distribution.

    Returns:
    --------
    m : float

    s : float

    """

    # Constant common to both variables.
    c = 1 + np.power(sig/mu,2)

    # Compute m and s.
    m = np.log(mu/np.sqrt(c))
    s = np.sqrt(np.log(c))

    return m,s
################################################################################



def posterior(theta, prior, ip_data):
    """
    Posterior density function.

    # NOTE: Could vectorise this to take an array of theta values.

    # NOTE 2: could be more efficient by computing posterior only if required.

    Parameters:
    -----------
    theta : float
        Unknown (logarithm of) thermal diffusivity.
    prior : array_like ([dist_type, param1, param2])
        Vector defining the prior distribution through the dist_type ('normal'
        or 'unif'), and two parameters (floats, mean and std for Gaussian, LHS
        and range for Uniform).
    ip_data : dictionary
        Parameters defining the inverse problem.

    Returns:
    --------
    posterior : float
        Value of the (un-normalised) posterior density evaluated at the input
        theta.
    prior : float
        Value of the prior density evaluated at the input theta.
    likelihood : float
        Value of the likelihood function evaluated at the input theta.
    """

    # Prior density.
    if prior[0] == "unif":
        p0 = Unif.pdf(theta, prior[1], prior[2]-prior[1])  # Uniform pdf
    elif prior[0] == "normal":
        p0 = Norm.pdf(theta, prior[1], prior[2]) # Gaussian pdf

    # Forward evaluation.
    F = uxt(ip_data['x_obs'], ip_data['t_obs'], np.exp(theta)).ravel()

    # Evaluate likelihood (multivariate Normal).
    L = mvNorm.pdf(F, ip_data['d'], ip_data['sig_rho'])

    # Return evaluation of the posterior, prior and likelihood.
    return p0*L, p0, L
################################################################################



def plot_posterior(prior, Nt=100):
    """
    Plot the true (analytic) posterior density function along with the likelihood,
    prior and true value of theta.

    Parameters:
    -----------
    prior : array_like ([dist_type, param1, param2])
        Vector defining the prior distribution through the dist_type ('normal'
        or 'unif'), and two parameters (floats, mean and std for Gaussian, LHS
        and range for Uniform).
    Nt : int
        Number of points at which to evaluate each density function.

    Returns:
    --------
    Plot the true (analytic) posterior density function along with the likelihood,
    prior and true value of theta.
    """

    # Theta values for evaluating the posterior.
    tv = np.linspace(0,3,Nt)

    # Allocate storage for likelihood and posterior evaluations.
    p0v = np.zeros(Nt)   # prior evaluations
    Lv = np.zeros(Nt)    # likelihood evaluations

    # Evaluate prior and likelihood.
    for n in range(Nt):
        #print(n)
        dummy_var, p0v[n], Lv[n] = posterior(tv[n], prior, ip_data)

    # Evaluate posterior.
    pv = p0v*Lv

    # Normalise.
    p0v = p0v/np.trapz(p0v,tv)
    Lv = Lv/np.trapz(Lv,tv)
    pv = pv/np.trapz(pv,tv)

    # Plot.
    fig, ax = myfigure()
    ax.plot(tv, pv, label='posterior')
    ax.plot(tv, Lv, label='likelihood')
    ax.plot(tv, p0v, label='prior')
    plt.axvline(x=np.log(ip_data['lam']), color='c', label='truth')
    ax.set_xlabel('theta')
    ax.set_ylabel('probability density')
    leg = ax.legend();
    plt.show(block=False)
################################################################################



def rwmh_posterior(N, NB, prior, ip_data, sig_q):
    """
    Sample from the posterior distribution using Random Walk Metropolis-Hastings
    MCMC.

    Parameters:
    -----------
    N : int
        Number of samples required.
    NB : int
        Number of burn-in samples.
    prior : array_like ([dist_type, param1, param2])
        Vector defining the prior distribution through the dist_type ('normal'
        or 'unif'), and two parameters (floats, mean and std for Gaussian, LHS
        and range for Uniform).
    ip_data : dictionary
        Parameters defining the inverse problem.
    sig_q : float
        Standard deviation of the proposal distribution.

    Returns:
    --------
    chain : array
        Samples from the posterior distribution.
    prop : float
        Proportion of accepted proposals.
    tvec : array
        Samples from posterior including burn-in.
    """

    # Storage for the sampled values of theta.
    tvec = np.zeros(N+NB)


    # Initialise counter for number of accepted samples.
    count = 0


    # Initial state
    tvec[0] = np.log(ip_data['lam'])  # true value

    # Evaluate posterior at initial state.
    post, dummy1, dummy2 = posterior(tvec[0],prior,ip_data)


    # Monte Carlo loop.
    for n in range(NB+N-1):

        # Generate proposal.
        t_p = Norm.rvs(tvec[n],sig_q)

        # Evaluate posterior at proposal.
        post_p, dummy1, dummy2 = posterior(t_p, prior, ip_data)

        # Compute acceptance probability.
        alpha = min(1, post_p/post)

        # Accept/reject step.
        z = Unif.rvs()
        if z < alpha:
            # Accept and update.
            tvec[n+1] = t_p
            post = post_p

            if n > NB:
                # Increase counter (post-burn in only).
                count = count + 1

        else:
            # Reject and update.
            tvec[n+1] = tvec[n]


        # Output progress updates to screen.
        if n < NB-2:
            if np.floor(100*(n+2)/NB) > np.floor(100*(n+1)/NB):
                print("Burning in, progress: " + str(n+2) + "/" + str(NB) + " samples computed, " + str(np.floor(100*(n+2)/NB)) + "% complete.")

        elif n == NB-2:
            print("Burn in complete.")

        elif n > NB-2:
            if np.floor(100*(n-NB+2)/N) > np.floor(100*(n-NB+1)/N):
                print("Progress: " + str(n-NB+2) + "/" + str(N) + " samples computed, " + str(np.floor(100*(n-NB+2)/N)) + "% complete.")


    # Proportion of proposals accepted.
    prop = count/N
    print("Proportion of proposals accepted is " + str(prop) +  ".")

    # Remove burn in samples to define chain.
    chain = tvec[NB:]

    # Output chain, proportion and full chain (with burn-in included).
    return chain, prop, tvec
################################################################################



def hist_plot(chain, nbins=100):
    """
    Plot a histogram of the input Markov chain.

    Parameters:
    -----------
    chain : array
        Samples from the posterior distribution.
    nbins = int
        Number of histogram bins.

    Returns:
    --------
    Plot a histogram of the input Markov chain.
    """

    # Compute histogram.
    hist, bin_edges = np.histogram(chain, nbins, density=1)

    # Compute bin centres.
    bin_centres = (bin_edges[1:] + bin_edges[:-1])/2


    # Plot histogram manually.
    fig, ax = myfigure()
    ax.plot(bin_centres, hist, label='posterior')
    plt.axvline(x=np.log(ip_data['lam']), color='r', label='truth')
    ax.set_xlabel('theta')
    ax.set_ylabel('probability density')
    leg = ax.legend();
    plt.show()
################################################################################




#####################################
###  ---  EXAMPLE EXECUTION  ---  ###
#####################################

if __name__ == "__main__":

    ## ----------------
    ## DATA COMPONENTS:
    ## ----------------

    # Unpickle the inverse problem data.
    pickle_in = open("ip_data.pickle","rb")
    ip_data = pickle.load(pickle_in)


    # Print dictionary to screen.
    for key in ip_data:
        print(key, ip_data[key])
################################################################################



    ## ----------
    ## PLOT DATA:
    ## ----------

    # Spatial locations (plotting).
    x = np.linspace(0,1,50)

    # Matrix representation of noisy data.
    d_mat = np.reshape(ip_data['d'],(ip_data['nt'],ip_data['nx']))

    # Labels for y-axes.
    ylabs = ['u(x,0)', 'u(x,T/3)','u(x,2T/3)','u(x,T)']

    # Initiate 2x2 plot.
    fig, ax = myfigure(nrows=2, ncols=2)

    for i,axi in enumerate(ax):

        if i == 0:
            # Initial data.
            axi.plot(x, np.squeeze(uxt(x,0,ip_data['lam'])))
        else:
            # Measurement times data.
            axi.plot(x, np.squeeze(uxt(x,ip_data['t_obs'][i-1],ip_data['lam'])))
            axi.plot(ip_data['x_obs'], d_mat[i-1,:], 'xr')

        axi.set_xlabel('x')
        axi.set_ylabel(ylabs[i])
        axi.axis([0, 1, 0, 3])

    plt.show(block=False)
    ############################################################################



    ## ---------------
    ## PLOT POSTERIOR:
    ## ---------------

    # Prior parameters
    m, s = m_s_from_mu_sig(4,1)
    prior = ["normal", m, s]

    print(m)
    print(s)

    plot_posterior(prior,200)
    ############################################################################



    # -----------------------
    # SAMPLE USING RWMH MCMC:
    # -----------------------

    NB = int(1e3)  # number of burn in samples
    N = int(5e4)   # number of samples

    nbins = 100  # number of histogram bins

    sig_q = 5.5e-1  # proposal standard deviation

    chain, prop, tvec = rwmh_posterior(N, NB, prior, ip_data, sig_q)
    ############################################################################



    # --------------
    # EXAMINE CHAIN:
    # --------------

    # Choose random set of points to plot.
    ran_ind = np.random.randint(1,N-1002)
    chain_plot = chain[ran_ind:ran_ind+1000]


    # Plot chosen section of chain.
    fig, ax = myfigure()
    ax.plot(range(ran_ind,ran_ind+1000), chain_plot)
    plt.axhline(y=np.log(ip_data['lam']), color='r', label='truth')
    ax.set_xlabel('iterate, n')
    ax.set_ylabel('theta')
    plt.show(block=False)
    ############################################################################



    # ---------------
    # PLOT HISTOGRAM:
    # ---------------


    # Plot histogram of chain using default number of bins.
    hist_plot(chain)
    ############################################################################





    print(ip_data['G'])
