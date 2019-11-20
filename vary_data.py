import util
import pickle
import numpy as np
import matplotlib.pyplot as plt
from util_fig import myfigure
from copy import deepcopy

# Unpickle the inverse problem data.
pickle_in = open("ip_data.pickle","rb")
ip_data = pickle.load(pickle_in)



## Single Time point.
ip_data2 = deepcopy(ip_data)

# Time.
ip_data2['nt'] = 1
ip_data2['t_obs'] = 2*ip_data['T']/3


# True observations.
ip_data2['G'] = ip_data['G'][4:8]
ip_data2['nd'] = ip_data2['nx']*ip_data2['nt']

# Noise.
ip_data2['rho_noise'] = ip_data['rho_noise'][4:8]

# Noisy data.
ip_data2['d'] = ip_data2['G'] + ip_data2['rho_noise']



## Single time point, single location.
ip_data3 = deepcopy(ip_data)

# Time.
ip_data3['nt'] = 1
ip_data3['t_obs'] = 2*ip_data['T']/3

# Space
ip_data3['nx'] = 1                                                   # number of spatial measurements
ip_data3['x_obs'] = 0.8   # observation locations



#print(ip_data['G'][8])

# True observations.
ip_data3['G'] = ip_data['G'][7]
ip_data3['nd'] = ip_data3['nx']*ip_data3['nt']

# Noise.
ip_data3['rho_noise'] = ip_data['rho_noise'][7]

# Noisy data.
ip_data3['d'] = ip_data3['G'] + ip_data3['rho_noise']





## Define prior.

m, s = util.m_s_from_mu_sig(4,1)
prior = ['normal', m, s]




# Number of evaluation points.
Nt = 200

# Theta values for evaluating the posteriors.
tv = np.linspace(0,3,Nt)

# Allocate storage for likelihood and posterior evaluations.
p0v = np.zeros(Nt)     # prior evaluations
pv = np.zeros((3,Nt))  # Posterior evaluations.

# Evaluate posterior.
for n in range(Nt):
    pv[0,n], p0v[n], dummy1 = util.posterior(tv[n], prior, ip_data)
    pv[1,n], dummy2,   dummy3 = util.posterior(tv[n], prior, ip_data2)
    pv[2,n], dummy4,   dummy5 = util.posterior(tv[n], prior, ip_data3)


# Normalise.
p0v = p0v/np.trapz(p0v,tv)
for i in range(3):
    pv[i] = pv[i]/np.trapz(pv[i],tv)


## PLOT

fig, ax = myfigure()
ax.plot(tv, pv[0],  'b', label='posterior1')
ax.plot(tv, pv[1],  'r', label='posterior2')
ax.plot(tv, pv[2],  'g', label='posterior3')
ax.plot(tv, p0v,   ':c', label='prior')

plt.axvline(x=np.log(ip_data['lam']), color='m', label='truth')

ax.set_xlabel('theta')
ax.set_ylabel('probability density')
leg = ax.legend();
plt.show()
