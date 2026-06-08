import torch 
from sbi.utils import BoxUniform
from standardization import standardize
import pandas as pd
import numpy as np

def generate_prior():
    """
    Generate the prior assuming the uniform distribution
    """
    # theta_train = np.array(pd.read_csv("./model 3/training_theta(poisson).csv", header=None))
    # x_train = np.array(pd.read_csv("./model 3/training_x(poisson).csv", header=None))

    # t, x = standardize(theta_train, x_train)
    

    low = torch.tensor([5, -1, -1, 0.2])
    high = torch.tensor([8, 0.7, 2, 1])
    
    # low = (low - t[1])/ t[2]
    # high = (high - t[1])/ t[2]
    
    # Create uniform distribution
    prior = BoxUniform(low=low, high=high)

    return prior