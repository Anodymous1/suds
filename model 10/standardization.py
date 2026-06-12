import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch 
import numpy as np
import pandas as pd

def standardize(theta_train, x_train):
    """
    Standardize the given datasets
    """
    theta_mean = np.mean(np.unique(theta_train, axis=0), axis=0)
    theta_std = np.std(np.unique(theta_train, axis=0), axis=0)

    theta_train = (theta_train - theta_mean)/ theta_std

    x_mean = np.mean(x_train, axis=0)
    x_std = np.std(x_train, axis=0)

    x_train = (x_train - x_mean)/ x_std

    theta_train = torch.from_numpy(theta_train).float()
    x_train = torch.from_numpy(x_train).float()
    
    return (theta_train, theta_mean, theta_std), (x_train, x_mean, x_std)

def get_standard():
    """
    get the standardized values of the data set
    """
    # theta = np.array(pd.read_csv("./model 3/training_theta(poisson).csv", header = None))
    # x = np.array(pd.read_csv("./model 3/training_x(poisson).csv", header = None))
    
    theta = np.array(pd.read_csv("./training_theta(poisson).csv", header = None))
    x = np.array(pd.read_csv("./training_x(poisson).csv", header = None))
    
    return standardize(theta, x)
    