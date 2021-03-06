
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
    lam : array_like
        Value of the thermal conductivity parameter.

    Returns:
    --------
    uxt : ndarray
        Value of the function u at each point in x and time in t for each given
        value of lambda.
    """

    # Ensure inputs are ndarrays.
    x = np.array(x)
    t = np.array(t)
    lam = np.array(lam)


    # Evaluate spatial component.
    X = 3*np.sin(np.pi*x) + np.sin(3*np.pi*x)


    # If lambda is scalar valued, compute a single outer product.
    if lam.ndim==0:
        T = np.array(np.exp(-lam * math.pow(np.pi,2) * t))
        U = np.outer(T,X)


    # If lambda is vector valued, stack a number of outer products.
    elif lam.ndim==1:

        # Evaluate temporal component for each lambda.
        T = np.exp(-math.pow(np.pi,2)*np.outer(lam,t))

        # Compute solution through stacking scaled outer products for each x.
        U = np.stack([T.T*i for i in X])

    return U
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
    posterior : nparray
        Value of the (un-normalised) posterior density evaluated at the input
        theta.
    prior : nparray
        Value of the prior density evaluated at the input theta.
    likelihood : nparray
        Value of the likelihood function evaluated at the input theta.
    """

     # Ensure input theta is nparray.
    theta = np.array(theta)


    # Prior density.
    if prior[0] == "unif":
        p0 = Unif.pdf(theta, prior[1], prior[2]-prior[1])  # Uniform pdf
    elif prior[0] == "normal":
        p0 = Norm.pdf(theta, prior[1], prior[2]) # Gaussian pdf


    # Single theta value.
    if theta.ndim == 0:
        F = uxt(ip_data['x_obs'], ip_data['t_obs'], np.exp(theta)).ravel()
        L = mvNorm.pdf(F, ip_data['d'], ip_data['sig_rho'])

    # Multiple theta values.
    elif theta.ndim == 1:
        F = uxt(ip_data['x_obs'], ip_data['t_obs'], np.exp(theta))
        F = F.reshape(-1, F.shape[2])

        L = np.stack([mvNorm.pdf(F[:,i], ip_data['d'], ip_data['sig_rho']) for i in range(theta.shape[0])])


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

    # Evaluate posterior, prior and likelihood.
    pv, p0v, Lv = posterior(tv, prior, ip_data)

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



def myhistogram(chain, nbins=None, bin_edges=None):
    """
    Compute a histogram of the input Markov chain.

    Parameters:
    -----------
    chain : array
        Samples from the posterior distribution.
    nbins = int
        Number of histogram bins.

    Returns:
    --------
    hist : array
        Probability density values.
    bin_centres : array
        Bin centre values.
    bin_edges : array
        Bin edge values.
    """

    # Compute histogram.
    if nbins is None:
        hist, bin_edges = np.histogram(chain, bins = bin_edges, density=1)
    elif bin_edges is None:
        hist, bin_edges = np.histogram(chain, bins = nbins, density=1)
    else:
        print("Error: You must specify either a number of bins or the bin edges.")
        return

    # Compute bin centres.
    bin_centres = (bin_edges[1:] + bin_edges[:-1])/2

    return hist, bin_centres, bin_edges
################################################################################



def hist_plot(hist, bin_centres):
    """
    Plot a histogram of the input Markov chain.

    Parameters:
    -----------
    hist : array
        Probability density values.
    bin_centres : array
        Bin centre values.

    Returns:
    --------
    Plot the input histogram.
    """

    # Plot histogram (manually).
    fig, ax = myfigure()
    ax.plot(bin_centres, hist, label='posterior')
    plt.axvline(x=np.log(ip_data['lam']), color='r', label='truth')
    ax.set_xlabel('theta')
    ax.set_ylabel('probability density')
    leg = ax.legend();
    plt.show()
################################################################################



def KL_div(x, p, q):
	"""
	Compute the Kullback-Leibler divergence
	D_{KL}(P||Q)
	between two distributions P and Q with densities p and q.

	Parameters:
	-----------
	x : array
	Points at which p and q are evaluated.
	p,q : arrays
		Evaluations of the densities of P and Q at the points x.

	Returns:
	--------
	The KL divergence between P and Q.
	"""

	#if np.count_nonzero(q) > 0:
	#    print('p must be absolutely continuous wrt q.')
	#    return

	# Evaluate integrand.
	#temp = p*np.log(p/q);

	# Deal with case where p(i)=0 (using fact xlogx = 0 in limit x-->0.
	#temp[np.isnan(temp)] = 0;
	#temp[np.isinf(temp)] = 0;

	temp = np.zeros(len(x))
	temp[np.where((p!=0) & (q!=0))] = p[np.where((p!=0) & (q!=0))]*np.log(p[np.where((p!=0) & (q!=0))]/q[np.where((p!=0) & (q!=0))])
	temp[np.where(p==0)] = 0
	temp[np.where(q==0)] = 0

	# Approximate integral.
	KL = np.trapz(temp, x);
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
    N = int(1e4)   # number of samples

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

    # Compute histogram.
    hist, bin_centres, bin_edges = myhistogram(chain, nbins=100)

    # Plot histogram of chain using default number of bins.
    hist_plot(hist, bin_centres)
    ############################################################################
