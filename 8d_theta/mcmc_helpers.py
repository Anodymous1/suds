from typing import Any, Callable, Optional, Tuple
import agama
import torch 
import numpy as np
from astropy import units as u
from sbi.inference import  MCMCPosterior, SNLE
from sbi.inference.potentials.likelihood_based_potential import LikelihoodBasedPotential
from sbi.utils import mcmc_transform
import torch
from torch import Tensor, nn
from torch.distributions import Distribution

from sbi.utils import mcmc_transform
from sbi.utils.torchutils import atleast_2d
from typing import Any, Callable, Optional, Tuple, Union

import torch
from arviz.data import InferenceData
from torch import Tensor

agama.setUnits(mass=1 * u.Msun, length=1*u.kpc, velocity=1 * u.km /u.s)


class LikelihoodBasedPotentialWithUncertainty(LikelihoodBasedPotential):
    def __init__(
        self,
        *args,
        uncertainties: Tensor = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.uncertainties = uncertainties
    
    def __call__(self, theta: Tensor, track_gradients: bool = True) -> Tensor:
        r"""Returns the potential $\log(p(x_o|\theta)p(\theta))$.

        Args:
            theta: The parameter set at which to evaluate the potential function. \
                uncertainties must be located at the final columns
            track_gradients: Whether to track the gradients.

        Returns:
            The potential $\log(p(x_o|\theta)p(\theta))$.
        """
        # Ensure theta is at least 2D
        theta = atleast_2d(theta)
        
        N = theta.shape[0]  # Number of candidate parameters
        M = self.x_o.shape[0]         # Number of trials
        
        # 1. Replicate theta candidates: [t0, t0, ..., t1, t1, ...] -> shape (N * M, D_theta)
        theta_rep = theta.repeat_interleave(M, dim=0)
        
        # 2. Replicate trial-specific uncertainties: [u0, u1, ..., u0, u1, ...] -> shape (N * M, D_unc)
        unc_rep = self.uncertainties.repeat(N, 1)
        
        # 3. Combine parameter candidate with the trial-specific uncertainty -> shape (N * M, D_theta + D_unc)
        theta_combo = torch.cat([theta_rep, unc_rep], dim=1)
        
        # 4. Replicate observed trials to match the combinations -> shape (N * M, D_x)
        # Keeps any extra dimensions if x_o is multidimensional
        x_rep = self.x_o.repeat(N, *([1] * (self.x_o.dim() - 1)))
        
        # Calculate likelihood over all xaN * M combinations in one batch
        with torch.set_grad_enabled(track_gradients):
            log_prob_batch = self.likelihood_estimator.log_prob(
                x_rep.to(self.device), 
                theta_combo.to(self.device)
            )
            
            # Reshape back to (N, M) and sum across the M trials for each of the N candidates
            log_likelihood_trial_sum = log_prob_batch.reshape(N, M).sum(dim=1)

        # Compute prior probabilities
        log_prior = self.prior.log_prob(theta)
        
        # Return as a scalar if an unbatched parameter was originally passed
        total_potential = log_likelihood_trial_sum + log_prior
        return total_potential.squeeze() if total_potential.numel() == 1 else total_potential
    
def likelihood_estimator_based_potential_with_uncertainty(
    likelihood_estimator: nn.Module,
    prior: Distribution,
    x_o: Optional[Tensor],
    uncertainties: Tensor,
    enable_transform: bool = True,
) -> Tuple[Callable, Any]:
    r"""Returns potential $\log(p(x_o|\theta)p(\theta))$ for likelihood-based methods.

    It also returns a transformation that can be used to transform the potential into
    unconstrained space.

    Args:
        likelihood_estimator: The neural network modelling the likelihood.
        prior: The prior distribution.
        x_o: The observed data at which to evaluate the likelihood.
        enable_transform: Whether to transform parameters to unconstrained space.
             When False, an identity transform will be returned for `theta_transform`.

    Returns:
        The potential function $p(x_o|\theta)p(\theta)$ and a transformation that maps
        to unconstrained space.
    """

    device = str(next(likelihood_estimator.parameters()).device)

    potential_fn = LikelihoodBasedPotentialWithUncertainty(
        likelihood_estimator, prior, x_o, uncertainties=uncertainties, device=device
    )
    theta_transform = mcmc_transform(
        prior, device=device, enable_transform=enable_transform
    )

    return potential_fn, theta_transform


class MCMCPosteriorWithUncertainty(MCMCPosterior):
    def sample(
        self,
        *args,
        uncertainty: Optional[Tensor] = None,
        **kwargs,
    ) -> Union[Tensor, Tuple[Tensor, InferenceData]]:
        
        if uncertainty is not None:
            self.potential_fn.uncertainties = uncertainty
            
        return super().sample(*args, **kwargs)
    
class CombinedLikelihoodEstimator():
    """
    A wrapper that combines the log prob function of two NLE models \
        where one model takes in 3D stellar kinematics while the other \
            takes in 5d
            
    Attributes:
        - net3: net that takes 3d stellar kinematics as input (x, y, vz)
        - net5: net that takes 5d stellar kinematics as input (x, y, vx, vy, vz)
    """
    
    def __init__(self, net3: SNLE, net5: SNLE):
        self.net3 = net3._neural_net
        self.net5 = net5._neural_net
    
    def log_prob(self, x:torch.Tensor, theta:torch.Tensor):
        """
        Return the combined log likelihood
        """
        
        mask = x[:, 3].isnan() & x[:, 4].isnan()
        three_d_x, five_d_x = x[mask], x[~mask]
        three_d_theta, five_d_theta = theta[mask], theta[~mask]
        
        return self.net3.log_prob(three_d_x, three_d_theta) + self.net5.log_prob(five_d_x, five_d_theta)
        