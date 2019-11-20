import util
import pickle
import numpy as np
import matplotlib.pyplot as plt
from util_fig import myfigure

# Unpickle the inverse problem data.
pickle_in = open("ip_data.pickle","rb")
ip_data = pickle.load(pickle_in)



## Define priors.

# Prior 1: Unif(0,3).
prior1 = ['unif', 0, 3]

# Prior 2: Norm(m,s^{2})
m, s = util.m_s_from_mu_sig(4,1)
prior2 = ['normal', m, s]

# Prior 3: Norm(theta^*, 0.1^{2})
prior3 = ['normal', np.log(ip_data['lam']), 0.1]



# Number of evaluation points.
Nt = 200

# Theta values for evaluating the posterior.
tv = np.linspace(0,3,Nt)

# Allocate storage for likelihood and posterior evaluations.
p0v = np.zeros((3,Nt))   # prior evaluations
pv = np.zeros((3,Nt))

# Evaluate posterior.
for n in range(Nt):
    pv[0,n], p0v[0,n], dummy1 = util.posterior(tv[n], prior1, ip_data)
    pv[1,n], p0v[1,n], dummy2 = util.posterior(tv[n], prior2, ip_data)
    pv[2,n], p0v[2,n], dummy3 = util.posterior(tv[n], prior3, ip_data)


# Normalise.
for i in range(3):
    p0v[i] = p0v[i]/np.trapz(p0v[i],tv)
    pv[i] = pv[i]/np.trapz(pv[i],tv)


## PLOT

fig, ax = myfigure()
ax.plot(tv,  pv[0], '-b', label='posterior1')
ax.plot(tv, p0v[0], ':b', label='prior1')

ax.plot(tv,  pv[1], '-r', label='posterior2')
ax.plot(tv, p0v[1], ':r', label='prior2')

ax.plot(tv,  pv[2], '-g', label='posterior3')
ax.plot(tv, p0v[2], ':g', label='prior3')

plt.axvline(x=np.log(ip_data['lam']), color='c', label='truth')

ax.set_xlabel('theta')
ax.set_ylabel('probability density')
leg = ax.legend();
plt.show()
