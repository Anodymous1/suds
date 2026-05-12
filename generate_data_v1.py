# Without the following work-around line, pytorch is incompatible with agama
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import agama
import numpy as np
import torch
from sbi.inference import NLE
from sbi.utils import BoxUniform
from sbi.inference import likelihood_estimator_based_potential
from sbi.analysis import conditional_pairplot

import pandas as pd

torch.manual_seed(13)
agama.setRandomSeed(13)
np.random.seed(13)

# Set units
agama.setUnits(mass=1, length=1, velocity=1)


def make_potential(p_0: float, r_s: float, gamma: float) -> agama.Potential:
    """
    Makes potential according to GNFW profile
    
    - p_0: density normalization
    - r_s: scale radius
    - gamma: inner slope

    Preconditions:
    -   0 <= gamma <= 1
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
        "type": "Spheroid",
        "mass": 1,
        "scaleRadius": r_star,
        "gamma": 0,
        "beta": 5,
        "alpha": 2,
    }
    
    return agama.Density(param)


def generate_galaxy(p_0: float, r_s: float, gamma: float, r_star: float, r_a: float):
    """
    Generate the galaxy model given theta

    - p_0: density normalization
    - r_s: scale radius
    - gamma: inner slope
    - r_star: scale length
    - r_a: the radius of transition from isotropic velocity orbits at small radii to radially biased orbits at larger radii (anisotropy radius).

    """

    pot = make_potential(p_0, r_s, gamma)

    df = agama.DistributionFunction(
        type = "QuasiSpherical",
        potential = pot,
        density = make_density(r_star),
        # anisType = "OsipkovMerritt",
        r_a = r_a
    )

    return agama.GalaxyModel(pot, df)

def transform_params(theta: torch.Tensor) -> torch.Tensor:
    """
    transform parameters into correct for generate_galaxy

    - theta: tensor of sampled theta with columns \
        log(p_0), log(r_s), gamma, r_star/r_s, r_a/r_star
    """


    p_0 = 10 ** theta[:,0]
    r_s = 10 ** theta[:,1]
    gamma = theta[:,2]
    r_star = theta[:,3] * r_s
    r_a = theta[:,4] * r_star

    return torch.stack([p_0, r_s, gamma, r_star, r_a], dim=1)


def generate_galaxy_bunch(theta: torch.Tensor) -> torch.Tensor:
    """
    Generate the galaxy model given theta

    - theta: tensor of sampled theta with columns \
        log(p_0), log(r_s), gamma, r_star/r_s, r_a/r_star
    """
    transformed_theta = transform_params(theta)
    
    samples_np = np.vstack([generate_galaxy(*row.tolist()).sample(1)[0][0] for row in transformed_theta])

    return torch.from_numpy(samples_np).to(torch.float32)  # sbi requires float 32

    
# Check table 1 from Nguyen et al. for prior values

# Create the prior boundary in following order: log(p_0), log(r_s), gamma, r_star/r_s, r_a/r_star
def generate_prior(n_samples: int):
    """
    Generate the thetas and x from uniform distribution of the prior
    """
    

    low = torch.tensor([5, -1, -1, 0.2, 0.5])
    high = torch.tensor([8, 0.7, 2, 1, 2])

    # Create uniform distribution
    prior = BoxUniform(low=low, high=high)

    # Sample from said distribution
    theta = prior.sample((n_samples,))

    # Generate the data
    x = generate_galaxy_bunch(theta)

    return prior, theta, x

prior, theta, x = generate_prior(10000)

theta = pd.DataFrame(theta.numpy())
x = pd.DataFrame(x.numpy())

theta.to_csv("./csv/theta.csv", header=False, index=False)
x.to_csv("./csv/x.csv", header=False, index=False)