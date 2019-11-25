import util
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
from util_fig import myfigure


## Data.

# Unpickle the inverse problem data.
pickle_in = open("ip_data.pickle","rb")
ip_data = pickle.load(pickle_in)
# -------------------------------------------------------------------------



## Prior

# Prior parameters
m, s = util.m_s_from_mu_sig(4,1)
prior = ["normal", m, s]
# -------------------------------------------------------------------------



## Compute true posterior.

# Theta values for evaluating the posterior.
Nt = int(5000)
bin_edges = np.linspace(0,3,num=Nt+1,endpoint=True)
bin_centres = (bin_edges[1:] + bin_edges[:-1])/2
tv = bin_centres

# Evaluate posterior, prior and likelihood.
pv, dummy1, dummy2 = util.posterior(tv, prior, ip_data)

# Normalise.
pv = pv/np.trapz(pv,tv)

# L2 norm of true posterior.
pL2 = np.sqrt(np.trapz(np.square(pv),tv))
# -------------------------------------------------------------------------



## Compute or unpickle full length chains.

compute_chain = 1

NB = int(1e4)  # number of burn in samples
N = int(1e7)   # number of samples

NC = int(5)

nbins = int(100)  # number of histogram bins

sig_q = 5.5e-1  # proposal standard deviation


# Compute chains if required
if compute_chain == 0:
	chains = np.zeros((N,NC)) # allocate storage for chains
	for i in range(NC):
		chains[:,i], dummy1, dummy2 = util.rwmh_posterior(N, NB, prior, ip_data, sig_q)

	
	pickle_out = open("chains_vary_sample.pickle","wb")
	pickle.dump(chains, pickle_out)
	pickle_out.close()

else:
	pickle_in = open("chains_vary_sample.pickle","rb")
	chains = pickle.load(pickle_in)
	print("Chains loaded")



NN = int(np.log10(N)-np.log10(100)+1)				   # number of N values to be used
Nv = np.ceil(np.logspace(np.log10(100), np.log10(N), NN))  # vector of N values



# allocate storage for histograms
hist = np.zeros((Nt,NC,NN))
errL2 = np.zeros((NC,NN))
errKL = np.zeros((NC,NN))
for nc in range(NC):
	for nn in range(NN):

		hist[:,nc,nn], dummy1, dummy2 = util.myhistogram(chains[0:int(Nv[nn]),nc],bin_edges=bin_edges)

		errL2[nc,nn] = np.sqrt(np.trapz(np.square(hist[:,nc,nn]-pv),tv))
		errKL[nc,nn] = util.KL_div(tv, hist[:,nc,nn], pv)

errL2_av = errL2.mean(axis=0)/pL2
errKL_av = errKL.mean(axis=0)


# Plot L2 errors
fig, ax = myfigure()
ax.loglog(Nv, errL2_av, label='L2 error')
ax.set_xlabel('N')
ax.set_ylabel('L2 error')
plt.show()


# Plot KL errors
fig, ax = myfigure()
ax.loglog(Nv, errKL_av, label='KL div')
ax.set_xlabel('N')
ax.set_ylabel('KL div')
plt.show()
	
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
