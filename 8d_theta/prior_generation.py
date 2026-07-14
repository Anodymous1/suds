import torch 
from sbi.utils import BoxUniform
from parameter_bounds import mins_with_uncertainty, mins_without_uncertainty, maxs_with_uncertainty, maxs_without_uncertainty


def generate_prior(uncertainty=False):
    """
    Generate the prior assuming the uniform distribution
    
    order of theta: 
    alpha, beta, gamma, log(p_0), log(r_s), log(r_star/r_s), log(r_a / r_star), beta_0
    """


    # Create uniform distribution
    if not uncertainty:
        prior = BoxUniform(low=mins_without_uncertainty, high=maxs_without_uncertainty)
    if uncertainty:
        prior = BoxUniform(low=mins_with_uncertainty, high=maxs_with_uncertainty)

    return prior