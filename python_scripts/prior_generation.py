import torch 
from sbi.utils import BoxUniform



def generate_prior():
    """
    Generate the prior assuming the uniform distribution
    """
    

    low = torch.tensor([5, -1, -1, 0.2])
    high = torch.tensor([8, 0.7, 2, 1])

    # Create uniform distribution
    prior = BoxUniform(low=low, high=high)

    return prior