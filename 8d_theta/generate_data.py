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
                  dim:int,
                  uncertainty:bool = False,
                  poisson:bool = True, 
                  n_jobs:int = 1) -> tuple[torch.Tensor]:
    """
    Generate the data in prepartion of saving as a csv file
    
    params:
    - num_galaxies: number of galaxies to generate
    - num_stars: the number of stars per galaxy, it is the mean of the poisson distribution if poisson=True
    - dim: dimension of x
    - uncertainty: to include uncertainty in the inference or not
    - poisson: use the poisson distribution to determine the number of stars to generate
    - n_jobs: number of CPU cores used to generate the data
    
    return:
    - x: the generate stars (3 dimensional, including x, y and v_los)
    - theta: the generate theta from a uniform distribution
    """
    
    prior = generate_prior()
    theta = prior.sample((num_galaxies,))
    
    if poisson:
        rates = torch.full((num_galaxies,), num_stars, dtype=torch.float32)
        n_stars = torch.poisson(rates).long()
    else:
        n_stars = num_stars

    if not uncertainty:
        theta = torch.column_stack((theta, n_stars))
        x = generate_galaxy_multiple(theta, n_stars, dim, n_jobs=n_jobs)
    else:
        theta = torch.repeat_interleave(theta, n_stars, dim=0)
        x, uncertainties = generate_galaxy_multiple(theta, n_stars, dim, uncertainty=True, n_jobs=n_jobs)
        theta = torch.column_stack((theta, uncertainties))
    
    return theta, x
    
def generate_single(theta:list[float], 
                    n_stars:int,
                    dim:int,
                    uncertainty:bool = False):
    """
    Generate a single galaxy
    
    Params:
    - theta: the parameters of the desired galaxies
    - n_stars: number of stars in the galaxy
    - dim: dimension of x
    - uncertainty: to include uncertainty in the inference or not
    """

    x = generate_galaxy_multiple(torch.tensor([theta]), torch.tensor([n_stars]), dim, uncertainty=uncertainty, n_jobs=1)
    return x
        

def compress(file:str, save_path:str):
    """
    Compress train_theta files, only works for data without uncertainty
    
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

def dimension_reduction(file:str, save_path:str, type:str):
    """
    Reduce the dimension of stellar kinematics, from 3 dimensional to 2 dimensional,\
        or from 5 dimensional to 3 dimensional
    The parameter R will be used in place of x and y
    
    Params:
    - file: file path to train_theta
    - save_path: where to save compressed file
    - type: type of file that needs to be compressed, either "x" or "theta" 
    """
    df = load_csv(file, "ndarray")
  
    if type == "x":
        if df.shape[1] == 3:
            new_array = np.zeros((df.shape[0], 2))
            new_array[:,0] = np.sqrt(df[:,0] ** 2 + df[:, 1] ** 2)
            new_array[:,1] = df[:,2]
        elif df.shape[1] == 5:
            new_array = df[:, (0, 1, 4)]
    elif type == "theta":
        if df.shape[1] == 11:
            new_array = df[:, :9]
            

    save_csv(new_array, save_path)
    

if __name__ == "__main__":
    # Generate Dataset
    # theta, x = generate_data(100,
    #                          100,
    #                          5,
    #                          uncertainty=True,
    #                          n_jobs=4)
    # for i in range(99):
    #     t, x0 = generate_data(100,
    #                          100,
    #                          5,
    #                          uncertainty=True,
    #                          n_jobs=4)
    #     theta = torch.cat((theta, t), dim=0)
    #     x = torch.cat((x, x0), dim=0)
    
    # save_csv(theta, "./8d_theta/model_7_1/5d/train_theta.csv", override=False)
    # save_csv(x, "./8d_theta/model_7_1/5d/train_x.csv", override=False)
    
    # # Single
    # x, _ = generate_single([1, 3, 0, 8.0755, 0, -0.6402, 0, 0, 0], 100, 5, uncertainty=True)
    # print(_)
    # save_csv(x, "./8d_theta/model_7/5d/mass_density_core.csv", override=True)
    
    # # Compress
    # compress("./8d_theta/model_2/train_theta.csv", "./8d_theta/model_2/train_theta_new.csv")
    
    # Reduce dimension
    dimension_reduction("./8d_theta/model_7_1/5d/train_x.csv", "./8d_theta/model_7_1/3d/train_x.csv", "x")
    