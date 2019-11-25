import numpy as np
import pickle


# Unpickle the inverse problem data.
pickle_in = open("ip_data.pickle","rb")
ip_data = pickle.load(pickle_in)
# -------------------------------------------------------------------------



## Compute true posterior.

# Theta values for evaluating the posterior.
Nt = 400
tv = np.linspace(0,3,Nt)

# Evaluate posterior, prior and likelihood.
pv, dummy1, dummy2 = posterior(tv, prior, ip_data)

# Normalise.
pv = pv/np.trapz(pv,tv)
# -------------------------------------------------------------------------



## Compute full length chain.

NB = int(1e3)  # number of burn in samples
N = int(1e4)   # number of samples

nbins = 100  # number of histogram bins

sig_q = 5.5e-1  # proposal standard deviation

chain, prop, tvec = rwmh_posterior(N, NB, prior, ip_data, sig_q)
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
