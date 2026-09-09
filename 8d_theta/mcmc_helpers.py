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

from scipy.stats import uniform, norm
import parameter_bounds as p

import torch
from arviz.data import InferenceData
from torch import Tensor

agama.setUnits(mass=1 * u.Msun, length=1*u.kpc, velocity=1 * u.km /u.s)


class LikelihoodBasedPotentialWithUncertainty(LikelihoodBasedPotential):
    def __init__(
        self,
        *args,
        uncertainties: Tensor = None,
        add_prior: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.uncertainties = uncertainties
        self.add_prior = add_prior
    
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
        total_potential = log_likelihood_trial_sum + log_prior if self.add_prior else log_likelihood_trial_sum
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
    
class RStarPrior():
    def __init__(self, r_star, r_star_unc):
        self.r_star_dist = norm(loc=r_star, scale=r_star_unc) if r_star_unc != 0 else r_star

    def generate_no_r_star_prior(self, log_r_star):
        # remove r_star from intervals
        loc = np.asarray(p.mins_without_uncertainty.copy())
        scale = np.asarray(p.maxs_without_uncertainty.copy()) - loc
        loc = np.delete(loc, 4)
        scale = np.delete(scale, 4)
        
        size = log_r_star.shape[0]
        loc = np.tile(loc, (size, 1))
        scale = np.tile(scale, (size, 1))
        loc[:, 4] -= log_r_star
        loc[:, 5] += log_r_star
        
        # create uniform distribution
        no_rstar_dist = uniform(loc=loc, scale=scale)
        
        return no_rstar_dist
        
    def logpdf(self, x):
        """
        Log prob
        
        Assumes that r_star is constant
        """
        
        log_r_star = x[:, 4] + x[:, 5]
        
        x = np.delete(x, 4, axis=1)
        x[:,4] -= log_r_star
        x[:,5] += log_r_star
        
        no_rstar_dist = self.generate_no_r_star_prior(log_r_star)
        
        if isinstance(self.r_star_dist, float):
            return np.sum(no_rstar_dist.logpdf(x), axis=1)
        else: 
            return np.sum(no_rstar_dist.logpdf(x), axis=1) + self.r_star_dist.logpdf(10 ** log_r_star)
    
    def rvs(self, size=1):
        """
        Sample
        
        params:
        - size: number of samples, must be scalar
        """
        # Sample r_star
        if isinstance(self.r_star_dist, float):
            log_r_star = np.log10(np.full(size, self.r_star_dist))
        else:
            log_r_star = np.log10(self.r_star_dist.rvs(size=size))
            
        
        # sample
        no_rstar_dist = self.generate_no_r_star_prior(log_r_star)
        no_rstar_samples = no_rstar_dist.rvs(size=(size,7))
        
        # Re-include r_star
        samples = np.zeros((size, 8))
        samples[:, :4] = no_rstar_samples[:, :4]
        samples[:, 4] = -no_rstar_samples[:, 4]
        samples[:, 5] = log_r_star + no_rstar_samples[:, 4]
        samples[:, 6] = no_rstar_samples[:, 5] - log_r_star
        samples[:, 7] = no_rstar_samples[:, 6]
        
        return samples
    
    @property
    def bounds(self):
        # min = np.asarray(p.mins_without_uncertainty.copy())
        # max = np.asarray(p.maxs_without_uncertainty.copy())
        
        # return np.column_stack((min, max))
        
        return np.column_stack((np.full((8, 1), -np.inf), np.full((8, 1), np.inf)))
    
    @property
    def dim(self):
        return 8
    
    def log_prob(self, *args):
        return self.logpdf(*args)
    
    def sample(self, *args):
        return self.rvs(*args)
    
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
