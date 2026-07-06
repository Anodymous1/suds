import os
os.environ["OMP_NUM_THREADS"] = "1"     # agama
os.environ["MKL_NUM_THREADS"] = "1"     # numpy, scipy
os.environ["OPENBLAS_NUM_THREADS"] = "1"    # numpy
os.environ["NUMEXPR_NUM_THREADS"] = "1"     # pandas

import torch 
import numpy as np

import pandas as pd
from galaxy_generation import generate_galaxy_multiple
from prior_generation import generate_prior
from object_handler import save_csv

torch.set_num_threads(1)


def generate_data(num_galaxies:int, 
                  num_stars:int, 
                  poisson:bool = True, 
                  n_jobs:int = 1) -> tuple[torch.Tensor]:
    """
    Generate the data
    
    params:
    - num_galaxies: number of galaxies to generate
    - num_stars: the number of stars per galaxy, it is the mean of the poisson distribution if poisson=True
    - poisson: use the poisson distribution to determine the number of stars to generate
    - n_jobs: number of CPU cores used to generate the data
    
    return:
    - x: the generate stars (3 dimensional, including x, y and v_los)
    - theta: the generate theta from a uniform distribution
    """
    
    prior = generate_prior()
    theta = prior.sample((num_galaxies,))
    
    if poisson:
        n_stars = torch.tensor(np.random.poisson(num_stars, size=num_galaxies))
    else:
        n_stars = num_stars


    theta = torch.repeat_interleave(theta, n_stars, dim=0)
    x = generate_galaxy_multiple(theta, n_stars, n_jobs)
    
    return theta, x
    
def generate_single(theta:list[float], 
                    n_stars:int):
    """
    Generate a single galaxy
    
    Params:
    - theta: the parameters of the desired galaxies
    - n_stars: number of stars in the galaxy
    """
    
    x = generate_galaxy_multiple(torch.tensor([theta]), [n_stars], 1)
    
    return x

if __name__ == "__main__":
    # theta, x = generate_data(1_000,
    #                          100,
    #                          n_jobs=4)
    # save_csv(theta, "./8d_theta/model_1/test_theta.csv")
    # save_csv(x, "./8d_theta/model_1/test_x.csv")
    
    # x = generate_single([1, 3, 1, 0, 8.0755, 0, -0.6402, 0, 0], 100)
    # save_csv(x, "./8d_theta/model_1/mass_density_cusp.csv")
    