import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch 
import numpy as np
import pandas as pd

def standardize(x_train: torch.Tensor) -> tuple[torch.Tensor]:
    """
    Standardize the given datasets
    
    returns the standardized data set with the mean and standard deviation
    
    params:
    
    """
    x_mean = torch.mean(x_train, dim=0)
    x_std = torch.std(x_train, dim=0, correction=0)

    x_train_std = (x_train - x_mean) / x_std
    
    return x_train_std.float(), x_mean.float(), x_std.float()
    