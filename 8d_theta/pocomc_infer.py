from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


from typing import Optional
import agama
import torch 
import numpy as np
import pandas as pd
from astropy import units as u
from sbi.inference import likelihood_estimator_based_potential, MCMCPosterior
from sbi.inference.potentials.likelihood_based_potential import LikelihoodBasedPotential, _log_likelihoods_over_trials
from sbi.analysis import conditional_potential
from sbi.utils import mcmc_transform
import time
from standardization import standardize
from object_handler import save_pickle, save_csv, load_csv, load_galaxies, load_pickle, load_h5
from joblib import Parallel, delayed
from prior_generation import generate_prior
from mcmc_helpers import likelihood_estimator_based_potential_with_uncertainty, MCMCPosteriorWithUncertainty, CombinedLikelihoodEstimator, LikelihoodBasedPotentialWithUncertainty
from mcmc import prep_data, save_samples
import parameter_bounds as p
from scipy.stats import uniform, norm
from mcmc_helpers import LikelihoodBasedPotentialWithUncertainty


# ================================================================================================================
import pocomc.tools
import pocomc.geometry
def patched_systematic_resample(n, weights):
    positions = (np.random.rand() + np.arange(n)) / n
    indices = np.zeros(n, dtype=int)
    cumulative_sum = weights[0]
    j = 0
    for i in range(n):
        # Added "and j < len(weights) - 1" safety guard to prevent j from exceeding bounds
        while positions[i] > cumulative_sum and j < len(weights) - 1:
            j += 1
            cumulative_sum += weights[j]
        indices[i] = j
    return indices

# ================================================================================================================

# set agama unit to be in Msun, kpc, km/s
agama.setUnits(mass=1 * u.Msun, length=1*u.kpc, velocity=1 * u.km /u.s)
from multiprocess import Pool

import pocomc as pc

agama.setRandomSeed(13)
torch.manual_seed(13)
np.random.seed(13)
torch.set_num_threads(1)



class RStarPrior():
    def __init__(self, r_star, r_star_unc):
        self.r_star_dist = norm(loc=r_star, scale=r_star_unc)

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
        
        return np.sum(no_rstar_dist.logpdf(x), axis=1) + self.r_star_dist.logpdf(10 ** log_r_star)
    
    def rvs(self, size=1):
        """
        Sample
        
        params:
        - size: number of samples, must be scalar
        """
        # Sample r_star
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
        


def create_potential(likelihood_estimator, prior, test_x, uncertainty):
    pot = LikelihoodBasedPotentialWithUncertainty(likelihood_estimator, prior, test_x, uncertainties=uncertainty, add_prior=False)
    
    return pot

def log_prob(t, potential):
    t = torch.tensor(t.astype(np.float32))
    with torch.no_grad():
        p = potential(t)
    return p.detach().cpu().numpy()

def create_prior():
    prior = pc.Prior([
        uniform(p.alpha_min, p.alpha_max - p.alpha_min),
        uniform(p.beta_min, p.beta_max - p.beta_min),
        uniform(p.gamma_min, p.gamma_max - p.gamma_min),
        uniform(p.log_rho_s_min, p.log_rho_s_max - p.log_rho_s_min),
        uniform(p.log_r_s_min, p.log_r_s_max - p.log_r_s_min),
        uniform(p.log_r_star_over_r_s_min, p.log_r_star_over_r_s_max - p.log_r_star_over_r_s_min),
        uniform(p.log_r_a_over_r_star_min, p.log_r_a_over_r_star_max - p.log_r_a_over_r_star_min),
        uniform(p.beta0_min, p.beta0_max - p.beta0_min),
    ])
    return prior


def sample_single_galaxy(i, likelihood_estimator, prior, x_o, uncertainty):
    import warnings
    warnings.filterwarnings("ignore", message="An x with a batch size of")
    warnings.filterwarnings("ignore", message="As of sbi v0.19.0")
    
    # --- APPLY MONKEYPATCH INSIDE WORKER ---
    import pocomc.tools
    import pocomc.geometry
    pocomc.tools.systematic_resample = patched_systematic_resample
    pocomc.geometry.systematic_resample = patched_systematic_resample
    # ---------------------------------------
    
    start_time = time.perf_counter()
    print(f"starting {i}th galaxy")
    
    pot = create_potential(likelihood_estimator, generate_prior(realistic_gamma=False), x_o, uncertainty)
    with Pool(2) as pool:
        sampler = pc.Sampler(
            prior = prior,
            likelihood=log_prob,
            likelihood_args=[pot],
            vectorize=True,
            random_state=13,
            # n_effective=1000,
            pool=pool)
        sampler.run()
    samples, logl, logp = sampler.posterior(resample=True)
    
    end_time = time.perf_counter()
    print(f"Galaxy {i} took {end_time - start_time:.4f} seconds")
    
    return samples


def run_mcmc(likelihood_estimator, prior, test_x, n_galaxies_at_once, uncertainty=None):
    
    final_samples = Parallel(n_jobs=n_galaxies_at_once, verbose=10)(
                delayed(sample_single_galaxy)(i, likelihood_estimator, prior, x_o, uncertainty[i]) 
                for i, x_o in enumerate(test_x)
            )
    return final_samples    
    




if __name__ == "__main__":
    
    # # MCMC on fixed rstar
    # mock = "B"
    # dim = 3
    # prof = "core"
    # test_x = prep_data(f"./8d_theta/model_8/mock/data/Mock{mock}_refined.csv",
    #                 train_x= "./8d_theta/model_8/5d/train_x.h5",
    #                 dim=3,)
    # uncertainty = torch.log10(load_csv(f"./8d_theta/model_8/mock/data/Mock{mock}_unc.csv", "Tensor"))
    # likelihood_estimator = load_pickle(f"./8d_theta/model_8/{dim}d/inference.pkl")._neural_net
    
    # prior = RStarPrior(0.22924, 0.004695)
    # samples = run_mcmc(likelihood_estimator, prior, test_x, 1, uncertainty=[uncertaint])
    
    # save_samples(samples, f"./8d_theta/model_8/mock/Mock{mock}_samples_fixed_t.csv")
    
# ======================================================================================================

    # # MCMC settings - P(v| x, y, sigma, theta)
                                        
    # Example code for mass density
    # prof = "cusp"
    mock = "A"
    dim = 3
    print(mock)
    test_x, position = prep_data(f"./8d_theta/model_8/mock/data/Mock{mock}_refined.csv",
                       train_x= f"./8d_theta/model_8/5d/train_x.h5",
                       uncertainty=True,
                       selection=True,
                       dim=dim)

    uncertainty = load_csv(f"./8d_theta/model_8/mock/data/Mock{mock}_unc.csv", "Tensor")

        
    likelihood_estimator = load_pickle(f"./8d_theta/model_9/{dim}d/inference.pkl")._neural_net
    
    prior = RStarPrior(0.22924, 0.004695)
    samples = run_mcmc(likelihood_estimator, prior, test_x, 1, uncertainty=[torch.column_stack((uncertainty, position))])
    
    
    save_samples(samples,
                 f"8d_theta/model_9/{dim}d/mock/Mock{mock}_samples.csv")
    
# ======================================================================================================
    # Model evaluation
    # dim = 3
    # test_x, uncertainties = prep_data(f"./8d_theta/model_8/{dim}d/test_x.h5",
    #                                   test_theta=f"./8d_theta/model_8/{dim}d/test_theta.h5",
    #                                   train_x=f"./8d_theta/model_8/5d/train_x.h5",
    #                                   dim=dim,
    #                                   uncertainty=True,
    #                                   num_entries=1000)
    # print(test_x[0].shape, uncertainties[0].shape)
    # print(test_x.__len__(), uncertainties.__len__())
    
    # likelihood_estimator = load_pickle(f"./8d_theta/model_8/{dim}d/inference.pkl")._neural_net
    
    # samples = run_mcmc(likelihood_estimator, create_prior(), test_x, 32, uncertainty=uncertainties)
    
    # save_samples(samples, f"./8d_theta/model_8/{dim}d/samples_poco.pkl")