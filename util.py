import math
import numpy as np
import matplotlib.pyplot as plt
from util_fig import myfigure        # plotting routine from Steven



def uxt(x,t,lam):
    """
    Evaluation of the function
        u(x,t;lambda) = (3sin(pi*x) + sin(3*pi*x))exp(-lambda*pi^2*t),
    which is the solution to the forward problem PDE.

    Parameters:
    -----------
    x : array_like
        spatial locations at which function is to be evaluated
    t : array_like
        time points at which function is to be evaluated
    lam : float
        value of the thermal conductivity parameter

    Returns:
    --------
    uxt : ndarray
        value of the function u at each point in x and time in t for the given value of lambda
    """

    # Spatial locations.
    X = np.array(3*np.sin(np.pi*x) + np.sin(3*np.pi*x))

    # Time points.
    T = np.array(np.exp(-lam * math.pow(np.pi,2) * t))

    # Compute solution through outer product and return values.
    return np.outer(T, X)



# Default example.
if __name__ == "__main__":
    ## DATA COMPONENTS:

    # Time.
    ip_data = {
        'T': 0.05,  # final time
        'nt': 3     # number of temporal measurements
    }
    ip_data['t_obs'] = ip_data['T']*np.arange(1,ip_data['nt']+1)/ip_data['nt']   # measurement times


    # Space
    ip_data['nx'] = 4                                                   # number of spatial measurements
    ip_data['x_obs'] = np.arange(1,ip_data['nx']+1)/(1+ip_data['nx'])   # observation locations


    # Spatial locations (plotting).
    x = np.linspace(0,1,50)


    # True observations.
    ip_data['lam'] = 5*np.pi/3                                                     # thermal conductivity
    ip_data['G'] = uxt(ip_data['x_obs'],ip_data['t_obs'],ip_data['lam']).ravel()   # true obs
    ip_data['nd'] = ip_data['nx']*ip_data['nt']                                    # number of obs


    # Noise.
    ip_data['seed'] = 0                                                           # seed RNG
    np.random.seed(0)
    ip_data['noise_ratio'] = 5/100                                                    # 5% signal-noise ratio
    ip_data['sig_rho'] = np.sqrt(ip_data['noise_ratio'] * np.mean(ip_data['G']))      # noise std
    ip_data['rho_noise'] = np.random.normal(0,ip_data['sig_rho'],(ip_data['nd'],1)).ravel()   # noise value


    # Noisy data.
    ip_data['d'] = ip_data['G'] + ip_data['rho_noise']


    # Print dictionary to screen.
    for key in ip_data:
        print(key, ip_data[key])


    ## PLOT DATA:

    # Matrix representation of noisy data.
    d_mat = np.reshape(ip_data["d"],(ip_data["nt"],ip_data["nx"]))

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

    plt.show()
