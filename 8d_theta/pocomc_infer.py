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
from scipy.stats import uniform, norm, rv_discrete
from mcmc_helpers import LikelihoodBasedPotentialWithUncertainty, RStarPrior
from memory_watchdog import start_memory_watchdog
# ================================================================================================================
import pocomc.tools
import pocomc.geometry
def patched_systematic_resample(n, weights):
    weights = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(weights)):
        raise ValueError(f"non-finite weights: {np.sum(~np.isfinite(weights))} of {len(weights)}")
    s = weights.sum()
    if abs(s - 1.0) > 1e-6:
        print(f"WARNING: weights sum to {s:.6f}, renormalizing")
    weights = weights / s

    positions = (np.random.rand() + np.arange(n)) / n
    indices = np.zeros(n, dtype=int)
    cumulative_sum = weights[0]
    j = 0
    clamped = 0
    for i in range(n):
        while positions[i] > cumulative_sum and j < len(weights) - 1:
            j += 1
            cumulative_sum += weights[j]
        if j == len(weights) - 1 and positions[i] > cumulative_sum:
            clamped += 1
        indices[i] = j
    if clamped:
        print(f"WARNING: {clamped}/{n} draws clamped to last particle")
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

        


def create_potential(likelihood_estimator, prior, test_x, uncertainty=False):
    if uncertainty is False:
        device = str(next(likelihood_estimator.parameters()).device)
        pot = LikelihoodBasedPotential(
        likelihood_estimator, prior, test_x, device=device
        )
    else:
        pot = LikelihoodBasedPotentialWithUncertainty(likelihood_estimator, prior, test_x, uncertainties=uncertainty, add_prior=False)
    
    return pot

def log_prob(t, potential):
    t = torch.tensor(t.astype(np.float32))
    with torch.no_grad():
        p = potential(t)
    return p.detach().cpu().numpy()

def create_prior(negative_beta0 = False):
    
    beta0_max = 0.0 if negative_beta0 else p.beta0_max
        
    prior = pc.Prior([
        uniform(p.alpha_min, p.alpha_max - p.alpha_min),
        uniform(p.beta_min, p.beta_max - p.beta_min),
        uniform(p.gamma_min, p.gamma_max - p.gamma_min),
        uniform(p.log_rho_s_min, p.log_rho_s_max - p.log_rho_s_min),
        uniform(p.log_r_s_min, p.log_r_s_max - p.log_r_s_min),
        uniform(p.log_r_star_over_r_s_min, p.log_r_star_over_r_s_max - p.log_r_star_over_r_s_min),
        uniform(p.log_r_a_over_r_star_min, p.log_r_a_over_r_star_max - p.log_r_a_over_r_star_min),
        uniform(p.beta0_min, beta0_max - p.beta0_min),
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
    
    pot = create_potential(
        likelihood_estimator,
        generate_prior(realistic_gamma=False),
        x_o,
        uncertainty=uncertainty,
    )

    sampler = pc.Sampler(
        prior=prior,
        likelihood=log_prob,
        likelihood_args=[pot],
        vectorize=True,
        random_state=13,
        n_effective=2048,
        n_active=512,
    )
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
    
    # MCMC on fixed rstar (mock)
    # start_memory_watchdog(limit_gb=40, min_available_gb=4, interval=1.0, log_every=5)
    
    # # mock = "D"
    # def infer(mock):
    #     # prof = "core"
    #     test_x = prep_data(f"./8d_theta/model_8/mock/data/Mock{mock}_refined.csv",
    #                     train_x= "./8d_theta/model_14/3d/train_x.h5",
    #                     dim=3,)
    #     uncertainty = torch.log10(load_csv(f"./8d_theta/model_8/mock/data/Mock{mock}_unc.csv", "Tensor"))
    #     likelihood_estimator = load_pickle(f"./8d_theta/model_14/3d/inference.pkl")._neural_net
        
    #     prior = RStarPrior(0.22924, 0.004695, negative_beta0=True)
    #     # prior = create_prior(negative_beta0=True)
    #     samples = run_mcmc(likelihood_estimator, prior, test_x, 1, uncertainty=[uncertainty])
        
    #     save_samples(samples, f"./8d_theta/model_14/3d/Mock{mock}_samples_beta0_p.csv")
    
    # Parallel(n_jobs=2, verbose=10)(
    #                 delayed(infer)(mock) 
    #                 for mock in "ABCD"
    #             )
    
# ======================================================================================================
    # MCMC on Real Data
    start_memory_watchdog(limit_gb=50, min_available_gb=4, interval=1.0, log_every=5)
    
    # R star
    galaxies = {
            # 'draco_1': (0.22924, 0.004695),
            'sculptor_1': (0.27268, 0.00513),
            'carina_1': (0.31003, 0.016035),
            'fornax_1': (0.82516, 0.01828),
            'sextans_1': (0.56865, 0.03137),
            # 'umi_1'
            }
    
    
    def infer(data, rstar):
        test_x = prep_data(f"./8d_theta/model_12/pace/ref_data/{data}.csv",
                        train_x= "./8d_theta/model_12/train_x.h5",
                        dim=3,)
        uncertainty = torch.log10(load_csv(f"./8d_theta/model_12/pace/ref_data/{data}_unc.csv", "Tensor"))
        likelihood_estimator = load_pickle(f"./8d_theta/model_12/inference.pkl")._neural_net
        
        prior = RStarPrior(*rstar, negative_beta0=True)
        samples = run_mcmc(likelihood_estimator, prior, test_x, 1, uncertainty=[uncertainty])
        
        save_samples(samples, f"./8d_theta/model_12/pace/{data}_samples.csv")
        
        
    # Parallel(n_jobs=1, verbose=10)(
    #                 delayed(infer)(data, r_star) 
    #                 for data, r_star in galaxies.items()
    #             )
    for data, r_star in galaxies.items():
        infer(data, r_star) 
    
# ======================================================================================================
    # # # MCMC on fixed rstar (mock) - No Uncertainty
    # mock = "A"
    # dim = 3
    # # prof = "core"
    # test_x = prep_data(f"./8d_theta/model_11/mock/data/Mock{mock}.csv",
    #                 train_x= "./8d_theta/model_11/train_x.h5",
    #                 dim=3,)
    # likelihood_estimator = load_pickle(f"./8d_theta/model_11/inference.pkl")._neural_net
    
    # prior = RStarPrior(0.22924, 0.004695)
    # samples = run_mcmc(likelihood_estimator, prior, test_x, 1, uncertainty=[None])
    
    # save_samples(samples, f"./8d_theta/model_11/Mock{mock}_samples_fixed.csv")
    
# ======================================================================================================
    
    # # MCMC on fixed rstar
    # dim = 3
    # prof = "cusp"
    # test_x = prep_data(f"./8d_theta/model_7_1/3d/mass_density_{prof}.csv",
    #                 train_x= "./8d_theta/model_8/5d/train_x.h5",
    #                 dim=3,)
    # # uncertainty = torch.log10(load_csv(f"./8d_theta/model_8/mock/data/Mock{mock}_unc.csv", "Tensor"))
    # uncertainty = torch.log10(torch.full((100,1), 1))
    
    # likelihood_estimator = load_pickle(f"./8d_theta/model_8/{dim}d/inference.pkl")._neural_net
    
    # prior = RStarPrior(0.229, 0)
    # samples = run_mcmc(likelihood_estimator, prior, test_x, 1, uncertainty=[uncertainty])
    
    # save_samples(samples, f"./8d_theta/model_8/3d/mass_density_samples_{prof}_fixed.csv")
    
# ======================================================================================================

    # # # MCMC settings - P(v| x, y, sigma, theta)
                                        
    # # Example code for mass density
    # # prof = "cusp"
    # mock = "A"
    # dim = 3
    # print(mock)
    # test_x, position = prep_data(f"./8d_theta/model_8/mock/data/Mock{mock}_refined.csv",
    #                    train_x= f"./8d_theta/model_8/5d/train_x.h5",
    #                    uncertainty=True,
    #                    selection=True,
    #                    dim=dim)

    # uncertainty = load_csv(f"./8d_theta/model_8/mock/data/Mock{mock}_unc.csv", "Tensor")

        
    # likelihood_estimator = load_pickle(f"./8d_theta/model_9/{dim}d/inference.pkl")._neural_net
    
    # prior = RStarPrior(0.22924, 0.004695)
    # samples = run_mcmc(likelihood_estimator, prior, test_x, 1, uncertainty=[torch.column_stack((uncertainty, position))])
    
    
    # save_samples(samples,
    #              f"8d_theta/model_9/{dim}d/mock/Mock{mock}_samples.csv")
    
# ======================================================================================================
    # # Model evaluation
    # dim = 3
    # test_x, uncertainties = prep_data(f"./8d_theta/model_12/test_x.h5",
    #                                   test_theta=f"./8d_theta/model_12/test_theta.h5",
    #                                   train_x=f"./8d_theta/model_12/train_x.h5",
    #                                   dim=dim,
    #                                   uncertainty=True,
    #                                   num_entries=2000)
    # # print(test_x[0].shape, uncertainties[0].shape)
    # # print(test_x.__len__(), uncertainties.__len__())
    
    # likelihood_estimator = load_pickle(f"./8d_theta/model_12/inference.pkl")._neural_net
    
    # samples = run_mcmc(likelihood_estimator, create_prior(), test_x, 32, uncertainty=uncertainties)
    
    # save_samples(samples, f"./8d_theta/model_12/samples_poco_2000_filtered.pkl")
    
# ======================================================================================================
    # # Model evaluation (No Uncertainty)
    # dim = 3
    # test_x, uncertainties = prep_data(f"./8d_theta/model_12/test_x.h5",
    #                                   test_theta=f"./8d_theta/model_12/test_theta.h5",
    #                                   train_x=f"./8d_theta/model_12/train_x.h5",
    #                                   dim=dim,
    #                                   uncertainty=True,
    #                                   num_entries=2000)
    # # print(test_x[0].shape, uncertainties[0].shape)
    # # print(test_x.__len__(), uncertainties.__len__())
    # # print(test_x.__len__(), test_x[0].shape)
    # likelihood_estimator = load_pickle(f"./8d_theta/model_12/inference.pkl")._neural_net
    
    # samples = run_mcmc(likelihood_estimator, create_prior(negative_beta0=True), test_x, 32, uncertainty=uncertainties)
    
    # save_samples(samples, f"./8d_theta/model_12/samples_poco_2000_beta0.pkl")
    