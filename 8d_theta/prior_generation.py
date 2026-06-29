import torch 
from sbi.utils import BoxUniform



def generate_prior():
    """
    Generate the prior assuming the uniform distribution
    
    order of theta: 
    alpha, beta, gamma, log(p_0), log(r_s), log(r_star/r_s), log(r_a / r_star), beta_0
    """
    

    low = torch.tensor([0.5, 2, -1, 3, -2, -3, -1, -0.5])
    high = torch.tensor([3, 10, 2, 10, 2, 0, 3, 1])

    # Create uniform distribution
    prior = BoxUniform(low=low, high=high)

    return prior