
import os
os.environ["OMP_NUM_THREADS"] = "1"     # agama
os.environ["MKL_NUM_THREADS"] = "1"     # numpy, scipy
os.environ["OPENBLAS_NUM_THREADS"] = "1"    # numpy
os.environ["NUMEXPR_NUM_THREADS"] = "1"     # pandas

import agama
import torch 
import numpy as np
from scipy import integrate
# from scipy.stats import wasserstein_distance_nd
from astropy import units as u

from sbi.utils import BoxUniform

from sbi.inference import SNLE, simulate_for_sbi, prepare_for_sbi


from sbi.utils import likelihood_nn

from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
import pickle
import matplotlib.pyplot as plt
from galaxy_generation import generate_galaxy_multiple, transform_params
from prior_generation import generate_prior
from standardization import standardize
from object_handler import load_csv


def density(x: np.ndarray, 
            theta: torch.Tensor,
            truth:bool = False) -> np.ndarray:
    """
    Calculate density using GNFW profile as a function of r. Returns the quantiles of the function
    
    Params:
    - x: the value at which the function must be computed at (log_r_div_rstar)
    - theta: MCMC samples of the posterior
    - truth: if True, return without quantiles
    """
    theta = np.array(transform_params(theta))
    
    alpha = theta[:,0]
    beta = theta[:,1]
    gamma = theta[:,2]
    p_0 = theta[:,3]
    r_s = theta[:,4]
    r_star = theta[:,5]
    
    out = np.zeros((theta.shape[0], x.shape[0]))
    for i, point in enumerate(x):
        r = 10 ** point * r_star
        out[:,i] = p_0 * (r / r_s) ** -gamma * (1 + (r / r_s) ** alpha) ** (-(beta-gamma)/alpha)
    
    if not truth:
        return np.quantile(np.log10(out), [0.025, 0.16, 0.50, 0.84, 0.975], axis=0).T
    else:
        return out


def density_single(r:float,
                   theta:torch.Tensor) -> np.ndarray:
    """
    Calculate density using GNFW profile as a function of r
    
    Parameters:
    - r: the value at which the function must be computed at
    - theta: an MCMC sample of the posterior
    
    Preconditions:
    - theta.shape == (8,)
    """
    theta = np.array(transform_params([theta]))

    alpha = theta[:,0]
    beta = theta[:,1]
    gamma = theta[:,2]
    p_0 = theta[:,3]
    r_s = theta[:,4]
    
    rho = p_0 * (r / r_s) ** -gamma * (1 + (r / r_s) ** alpha) ** (-(beta-gamma)/alpha)
    
    return rho

def betaCOM(x: float, 
            theta: torch.Tensor,
            truth: bool = False) -> np.ndarray:
    """
    Calculate betaCOM, Cuddeford-Osipkov- Merritt (COM) profile, as a function of r
    
    Params:
    - log_r_div_rstar: the value at which the function must be computed at
    - theta: MCMC samples of the posterior
    - truth: if True, return without quantiles
    """
    theta = np.array(transform_params(theta))
    
    r_star = theta[:,5]
    r_a = theta[:,6]
    beta_0 = theta[:,7]
    
    out = np.zeros((theta.shape[0], x.shape[0]))
    for i, point in enumerate(x):
        r = 10 ** point * r_star
        out[:,i] = (beta_0 + (r/r_a) ** 2 )/(1 + (r/r_a) ** 2)
    
    if not truth:
        return np.quantile(out, [0.025, 0.16, 0.50, 0.84, 0.975], axis=0).T
    else:
        return out


def mass(x: np.ndarray, 
         samples: torch.Tensor,
         truth: bool = False) -> np.ndarray:
    """
    Computes the mass for samples at each point of x
    
    Params:
    - x: linspace of the x-values that must be computed
    - samples: MCMC samples of the posterior
    - truth: if True, return without quantiles
    """
    
    out = np.zeros((samples.shape[0], x.shape[0]))
    for i, row in enumerate(samples):
        r = 10 ** x * (row[3] * 10 ** row[1])
        y = 4 * np.pi * r * r * density_single(r, row)
        
        M_0 = integrate.quad(lambda t: 4 * np.pi * t * t * density_single(t, row), 0, r[0])[0]
        I = integrate.cumulative_trapezoid(y, r) + M_0
        out[i,0] = M_0
        out[i,1:] = I
    
    if not truth:
        return np.quantile(np.log10(out), [0.025, 0.16, 0.50, 0.84, 0.975], axis=0).T
    else:
        return out


def prep_truth(x:np.ndarray, 
               theta: list[float]) -> tuple[np.ndarray]:
    """
    Prepare the true function
    
    Params: 
    - x: linspace of the x-values that must be computed
    - theta: true parameters of the equation
    """
    true_mass = mass(x, torch.tensor(theta)).squeeze()
    true_density = density(x, torch.tensor(theta)).squeeze()
    true_beta = betaCOM(x, torch.tensor(theta)).squeeze()
    
    return true_mass, true_density, true_beta


def plot_line(ax, x, y, truth, iloc):
    """
    plot the line graph
    
    params:
    - ax: the axis to plot on
    - x: x values
    - y: y values, must be in quantiles, y.shape == (n,5)
    - truth: truth values to plot, truth.shape == (n,)
    - iloc: location to plot, in indices
    """
    row, col = iloc
    #2.5-97.5
    ax[row, col].fill_between(x, y[:,0], y[:,4], color=(133/255, 133/255, 247/255, 0.5), label="Mid 95%")

    #16-84
    ax[row, col].fill_between(x, y[:,1], y[:,3], color=(133/255, 133/255, 247/255, 0.8), label="Mid 68%")

    # Median
    ax[row, col].plot(x, y[:,2], color=(8/255, 2/255, 134/255), label="Median")

    # Truth
    ax[row, col].plot(x, truth, color="red", linestyle = "--", label="Truth")
    
    ax[row, col].grid(True)
    ax[row, col].minorticks_on()
    
    return ax
    
    
def initialize_plot():
    plt.rcParams["text.usetex"] = True
    fig, axs = plt.subplots(3, 2, 
                            figsize=(8, 12),
                            sharex="col", 
                            sharey="row",
                            gridspec_kw={'hspace': 0,
                                            'wspace': 0,})

    xlim = 2
    resolution = 1000
    x = np.linspace(-xlim, xlim, resolution) # log_r_div_r_star
    
    return fig, axs, x

def plot_aesthetics(ax, fig):
    # Aesthetics
    
    # Core density
    ax[0,0].set_ylabel(r"$\log_{10} \rho (r) \ [M_\odot / \mathrm{kpc}^3]$")
    ax[0,0].xaxis.set_label_position('top')
    ax[0,0].set_xlabel(r"Cored Profile: $\gamma = 0$", fontweight="bold")
    
    # core mass
    ax[1,0].set_ylabel(r"$\log_{10} M(r) / M_\odot$")
    
    # Core beta
    ax[2,0].set_xlabel(r"$\log _{10} (r/r_*)$")
    ax[2,0].set_ylabel(r"$\beta ^\text{COM} (r)$")
    
    # Cusp density
    ax[0,1].xaxis.set_label_position('top')
    ax[0,1].set_xlabel(r"Cuspy Profile: $\gamma = 1$", fontweight="bold")
    ax[0,1].legend()
    
    # Cusp mass 
    
    # Cusp beta
    ax[2,1].set_xlabel(r"$\log _{10} (r/r_*)$")
    
    fig.suptitle("Recovered Density and Mass Profile of a Sample Galaxy (Model 14, 1k stars)" + "\n$\log_{10}(\\rho _0)=7,\ \log_{10}(r_s) = 0,\ r_* / r_s = 0.2$")

    fig.tight_layout()
    
    return ax, fig
    

if __name__ == "__main__":
    
    core_params = load_csv("./8d_theta/model_1/mass_density_samples_core.csv", "Tensor")
    cusp_params = load_csv("./8d_theta/model_1/mass_density_samples_cusp.csv", "Tensor")
    
    fig, ax, x = initialize_plot()
    
    core_density = density(x, core_params)
    core_mass = mass(x, core_params)
    core_beta = betaCOM(x, core_params)
    
    
    cusp_density = density(x, cusp_params)
    cusp_mass = mass(x, cusp_params)
    cusp_beta = betaCOM(x, cusp_params)
    
    true_cusp_density, true_cusp_mass, true_cusp_beta = prep_truth(x, [1, 3, 1, 0, 8.0755, 0, -0.6402, 0, 0])
    true_core_density, true_core_mass, true_core_beta = prep_truth(x, [1, 3, 0, 0, 8.0755, 0, -0.6402, 0, 0])
    
    ax = plot_line(ax, x, core_density, [0,0])
    ax = plot_line(ax, x, core_mass,    [1,0])
    ax = plot_line(ax, x, core_beta,    [2,0])
    ax = plot_line(ax, x, cusp_density, [0,1])
    ax = plot_line(ax, x, cusp_mass,    [1,1])
    ax = plot_line(ax, x, cusp_beta,    [2,1])
    
    ax, fig = plot_aesthetics(ax, fig)
    
    plt.show()
    