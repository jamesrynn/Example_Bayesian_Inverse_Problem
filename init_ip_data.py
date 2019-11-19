"""
Define inverse problem through parameters in a dictionary and pickle the
dictionary for future use.
"""

import numpy as np
from util import uxt
import pickle


# Time.
ip_data = {
    'T': 0.05,  # final time
    'nt': 3     # number of temporal measurements
}
ip_data['t_obs'] = ip_data['T']*np.arange(1,ip_data['nt']+1)/ip_data['nt']   # measurement times


# Space
ip_data['nx'] = 4                                                   # number of spatial measurements
ip_data['x_obs'] = np.arange(1,ip_data['nx']+1)/(1+ip_data['nx'])   # observation locations


# True observations.
ip_data['lam'] = 5*np.pi/3                                                     # thermal conductivity
ip_data['G'] = uxt(ip_data['x_obs'],ip_data['t_obs'],ip_data['lam']).ravel()   # true obs
ip_data['nd'] = ip_data['nx']*ip_data['nt']                                    # number of obs


# Noise.
ip_data['seed'] = 8992                                                           # seed RNG
np.random.seed(ip_data['seed'])
ip_data['noise_ratio'] = 2/100                                                    # 5% signal-noise ratio
ip_data['sig_rho'] = np.sqrt(ip_data['noise_ratio'] * np.mean(ip_data['G']))      # noise std
ip_data['rho_noise'] = np.random.normal(0,ip_data['sig_rho'],(ip_data['nd'],1)).ravel()   # noise value


# Noisy data.
ip_data['d'] = ip_data['G'] + ip_data['rho_noise']



pickle_out = open("ip_data.pickle","wb")
pickle.dump(ip_data, pickle_out)
pickle_out.close()
