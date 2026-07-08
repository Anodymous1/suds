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
from object_handler import save_csv, load_csv

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
        n_stars = np.random.poisson(num_stars, size=num_galaxies)
    else:
        n_stars = num_stars

    theta = torch.repeat_interleave(theta, torch.Tensor(n_stars).long(), dim=0)
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

def compress(file:str, save_path:str):
    """
    Compress train_theta files
    
    Params:
    - file: file path to train_theta
    - save_path: where to save compressed file
    """
    # load data
    df = load_csv(file, "ndarray")
    
    # Compress in np
    _, i = np.unique(df, axis=0, return_index=True)
    index = np.sort(i)
    new_df = df[index]
    
    # add the number of stars to the end 
    num = np.diff(np.concatenate((index, np.array([df.shape[0]]))))
    a = np.column_stack((new_df,num))
    
    # save
    save_csv(a, save_path)
    

if __name__ == "__main__":
    # Generate Dataset
    theta, x = generate_data(100,
                             1000,
                             n_jobs=4)
    for i in range(99):
        t, x0 = generate_data(100,
                             1000,
                             n_jobs=4)
        theta = torch.cat((theta, t), dim=0)
        x = torch.cat((x, x0), dim=0)
    
    save_csv(theta, "./8d_theta/model_3/train_theta.csv", override=True)
    save_csv(x, "./8d_theta/model_3/train_x.csv", override=True)
    
    # # Single
    # x = generate_single([1, 3, 1, 0, 8.0755, 0, -0.6402, 0, 0], 100)
    # save_csv(x, "./8d_theta/model_1/mass_density_cusp.csv")
    