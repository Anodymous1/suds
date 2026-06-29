import os
os.environ["OMP_NUM_THREADS"] = "1"     # agama
os.environ["MKL_NUM_THREADS"] = "1"     # numpy, scipy
os.environ["OPENBLAS_NUM_THREADS"] = "1"    # numpy
os.environ["NUMEXPR_NUM_THREADS"] = "1"     # pandas

import agama
import torch 
import numpy as np
from astropy import units as u
from joblib import Parallel, delayed
# set agama unit to be in Msun, kpc, km/s
agama.setUnits(mass=1 * u.Msun, length=1*u.kpc, velocity=1 * u.km /u.s)
agama.setRandomSeed(13)
torch.manual_seed(13)
np.random.seed(13)

def make_potential(p_0: float, r_s: float, gamma: float) -> agama.Potential:
    """
    Makes potential according to GNFW profile
    
    - p_0: density normalization
    - r_s: scale radius
    - gamma: inner slope

    """

    # Based on GNFW Profile
    param = {
        "type": "Spheroid", 
        "densityNorm": p_0,
        "scaleRadius": r_s,
        "gamma": gamma,
        "beta": 3,
        "alpha": 1
    }

    return agama.Potential(param)

def make_density(r_star: float):
    """
    Creates stellar density distribution according to the 3D Plummer Profile

    - r_star: scale length
    """

    # Based on Plummer profile
    param = {
        "type": "Plummer",
        "mass": 1,
        "scaleRadius": r_star,
    }
    
    return agama.Density(param)


def generate_galaxy(p_0: float, r_s: float, gamma: float, r_star: float):
    """
    Generate the galaxy model given theta

    - p_0: density normalization
    - r_s: scale radius
    - gamma: inner slope
    - r_star: scale length

    """

    pot = make_potential(p_0, r_s, gamma)

    df = agama.DistributionFunction(
        type = "QuasiSpherical",
        potential = pot,
        density = make_density(r_star),
        beta0 = 0.0,
        r_a = np.inf
    )

    return agama.GalaxyModel(pot, df)

def transform_params(theta: torch.Tensor) -> torch.Tensor:
    """
    transform parameters into correct for generate_galaxy

    - theta: tensor of sampled theta with columns \
        log(p_0), log(r_s), gamma, r_star/r_s
    """


    p_0 = 10 ** theta[:,0]
    r_s = 10 ** theta[:,1]
    gamma = theta[:,2]
    r_star = theta[:,3] * r_s

    return torch.stack([p_0, r_s, gamma, r_star], dim=1)

def simulate_one_galaxy(theta: np.ndarray, num_stars: int) -> np.ndarray:
    """
    Simulate one galaxy
    """
    model = generate_galaxy(*theta)
    stars, _ = model.sample(int(num_stars))
    return stars


def generate_galaxy_multiple(theta: torch.Tensor, n_stars: np.ndarray, n_jobs: int, d =2) -> torch.Tensor:
    """
    Generate the galaxy model with multiple stars given theta 

    returns a matrix of stars for each theta

    - theta: tensor of sampled theta with columns \
        log(p_0), log(r_s), gamma, r_star/r_s
    """
    transformed_theta = transform_params(theta)
    
    torch.set_num_threads(1)
    
    
    results = Parallel(n_jobs=n_jobs)(
    delayed(simulate_one_galaxy)(row, n) 
    for row, n in zip(transformed_theta, n_stars)
    )
    
    samples_np = np.vstack(results)
    
    if d == 2:
        out = np.zeros((samples_np.shape[0],2))
        out[:,0] = np.sqrt(samples_np[:,0] ** 2 + samples_np[:,1] ** 2)
        out[:,1] = samples_np[:, -1]
    elif d == 3:
        out = np.zeros((samples_np.shape[0],3))
        out[:,0] = samples_np[:, 0]
        out[:,1] = samples_np[:, 1]
        out[:,2] = samples_np[:, -1]
    
    return torch.from_numpy(out).to(torch.float32)  # sbi requires float 32