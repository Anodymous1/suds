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

def _make_potential(alpha: float, beta: float, gamma: float, p_0: float, r_s: float) -> agama.Potential:
    """
    Makes potential according to GNFW profile
    
    - alpha: transition sharpness
    - beta: outer slope
    - gamma: inner slope
    - p_0: density normalization
    - r_s: scale radius

    """

    # Based on GNFW Profile
    param = {
        "type": "Spheroid", 
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "densityNorm": p_0,
        "scaleRadius": r_s,
        "outerCutoffRadius": max(50, 10 * r_s)
    }

    return agama.Potential(param)

def _make_density(r_star: float):
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


def _generate_galaxy(alpha:float, 
                    beta:float, 
                    gamma:float, 
                    p_0: float, 
                    r_s: float, 
                    r_star: float,
                    r_a: float,
                    beta_0: float):
    """
    Generate the galaxy model given theta

    - alpha: Transition sharpness
    - beta: outer slope
    - gamma: inner slope 
    - p_0: density normalization
    - r_s: scale radius
    - r_star: stellar density scale radius
    - r_a: Anisotropy scale radius
    - beta_0: Central anisotropy

    """

    pot = _make_potential(alpha, beta, gamma, p_0, r_s)

    df = agama.DistributionFunction(
        type = "QuasiSpherical",
        potential = pot,
        density = _make_density(r_star),
        r_a = r_a,
        beta0 = beta_0,
    )

    return agama.GalaxyModel(pot, df)

def transform_params(theta: torch.Tensor) -> torch.Tensor:
    """
    transform parameters into correct for generate_galaxy

    - theta: tensor of sampled theta with columns \
        alpha, beta, gamma, log(p_0), log(r_s), log(r_star/r_s), log(r_a / r_star), beta_0
    """
    
    alpha = theta[:,0]
    beta = theta[:,1]
    gamma = theta[:,2]
    p_0 = 10 ** theta[:,3]
    r_s = 10 ** theta[:,4]
    r_star = 10 ** theta[:,5] * r_s
    r_a = 10 ** theta[:,6] * r_star
    beta_0 = theta[:,7]

    return torch.stack([alpha, beta, gamma, p_0, r_s, r_star, r_a, beta_0], dim=1)

def _sample_galaxy(model, num_stars: int) -> np.ndarray:
    """
    Simulate one galaxy and sample from it
    
    returns the generated stars before filtering
    
    params:
    - model: the galaxy model already set up
    - num_stars: number of stars in the galaxy
    
    """
    
    star_list = []
    remainder = num_stars % 100
    chunks = num_stars // 100
    
    if remainder > 0:
        stars, _ = model.sample(remainder)
        star_list.append(stars)

    for i in range(chunks):
        stars, _ = model.sample(100)
        star_list.append(stars)
    
    galaxy = np.vstack(star_list)
    return galaxy


def _simulate_one_galaxy(theta: torch.Tensor, num_stars: int) -> torch.Tensor:
    """
    Simulate one galaxy, then filtering the stars
    
    returns the generated (filtered) stars
    
    params:
    - theta: a single theta
    - num_stars: number of stars in the galaxy
    
    """
    theta = theta.tolist()
    model = _generate_galaxy(*theta)
    galaxy = _sample_galaxy(model, num_stars)

    r_star = theta[5]
    
    #  filter out v_los > 1000, r > 20 * r_star
    condition = (galaxy[:,5] > 1000) | (np.sqrt(galaxy[:,0] ** 2 + galaxy[:,1] ** 2) > 20 * r_star)
    while np.count_nonzero(condition) != 0:
        new_star = _sample_galaxy(model, np.count_nonzero(condition))
        galaxy[condition] = new_star
        condition = (galaxy[:,5] > 1000) | (np.sqrt(galaxy[:,0] ** 2 + galaxy[:,1] ** 2) > 20 * r_star)

    return torch.tensor(galaxy).float()



def generate_galaxy_multiple(theta: torch.Tensor, n_stars: np.ndarray, n_jobs: int) -> torch.Tensor:
    """
    Generate the galaxy model with multiple stars given theta 

    returns a matrix of stars for each theta

    - theta: tensor of sampled theta with columns \
        alpha, beta, gamma, log(p_0), log(r_s), log(r_star/r_s), log(r_a / r_star), beta_0
    - n_stars: the number of stars to generate for each corresponding theta
    - n_jobs: the number of threads to use to use in the generation process
    """
    transformed_theta = transform_params(theta)
    
    # torch.set_num_threads(1)
    # agama.setUnits(mass=1 * u.Msun, length=1*u.kpc, velocity=1 * u.km /u.s)
    # agama.setRandomSeed(13)
    # torch.manual_seed(13)
    # np.random.seed(13)
    
    results = Parallel(n_jobs=n_jobs)(
    delayed(_simulate_one_galaxy)(row, n) 
    for row, n in zip(transformed_theta, n_stars)
    )
    
    samples_np = torch.cat(results, dim=0)
    
    out = torch.zeros((samples_np.shape[0],3))
    out[:,0] = samples_np[:, 0]
    out[:,1] = samples_np[:, 1]
    out[:,2] = samples_np[:, -1]
    
    return out  # sbi requires float 32