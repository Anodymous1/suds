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
    Standardize the given datasets using PyTorch
    """
    # 1. Standardize theta_train based on its unique entries
    # np.unique(..., axis=0) -> torch.unique(..., dim=0)
    theta_unique = torch.unique(theta_train, dim=0)
    
    theta_mean = torch.mean(theta_unique, dim=0)
    
    # NumPy defaults to population standard deviation (ddof=0).
    # PyTorch defaults to sample standard deviation (correction=1).
    # Setting correction=0 keeps the math identical to NumPy's default.
    theta_std = torch.std(theta_unique, dim=0, correction=0)

    theta_train_std = (theta_train - theta_mean) / theta_std

    # 2. Standardize x_train
    x_mean = torch.mean(x_train, dim=0)
    x_std = torch.std(x_train, dim=0, correction=0)

    x_train_std = (x_train - x_mean) / x_std

    # 3. Cast everything to float32 (matches the original .float() conversion)
    return (
        (theta_train_std.float(), theta_mean.float(), theta_std.float()),
        (x_train_std.float(), x_mean.float(), x_std.float())
    )

def get_standard():
    """
    get the standardized values of the data set
    """
    theta = np.array(pd.read_csv("./model_11/train_theta.csv", header = None))
    x = np.array(pd.read_csv("./model_11/train_x.csv", header = None))
    
    return standardize(theta, x)

def un_standard(x_data):
    
    t, x = get_standard()
    return x_data * x[2] + x[1]
    