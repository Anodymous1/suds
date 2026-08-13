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
from mcmc_helpers import likelihood_estimator_based_potential_with_uncertainty, MCMCPosteriorWithUncertainty, CombinedLikelihoodEstimator
# set agama unit to be in Msun, kpc, km/s
agama.setUnits(mass=1 * u.Msun, length=1*u.kpc, velocity=1 * u.km /u.s)

agama.setRandomSeed(13)
torch.manual_seed(13)
np.random.seed(13)


def prep_data(test_x: str,
              test_theta: str|None = None,
              train_x: str|None = None,
              num_entries:int = 10000,
              uncertainty:bool = False,
              selection:bool = False,
              dim:int = 5) -> list[torch.Tensor]:
    """ 
    Prepare the data for MCMC
    
    params:
    - test_x: file location for test data of x
    - test_theta: file location for test data of theta. None if MCMC is ran on 1 galaxy
    - train_x: file location of the training x. None if standardization is not needed
    - num_entries: only keep the first num_entries test data. Only relevant for multiple galaxies
    - uncertainty: whether to include uncertainty in our inference or not. Only relevant for multiple galaxies
    - selection: if True, P(v| x, y, sigma, theta). Only relevant for multiple galaxies
    - dim: dimension of the stellar kinematic of train_x
    """
    
    test_x_raw = load_h5(test_x, "x", "Tensor") if ".h5" in test_x else load_csv(test_x, "Tensor")
    if test_x_raw.shape[1] != dim:
        if dim == 4:
            test_x_raw = test_x_raw[:, :4]
        elif dim == 3:
            test_x_raw = test_x_raw[:, (0, 1, 4)]
    
    # standardization
    if train_x is not None:
        train_x = load_h5(train_x, "x", "Tensor") if ".h5" in train_x else load_csv(train_x, "Tensor")
        if dim == 4:
            train_x = train_x[:, :4]
        elif dim == 3:
            train_x = train_x[:, (0, 1, 4)]
        
        x = standardize(train_x)
        test_x_raw = (test_x_raw - x[1])/ x[2]
    
    
    # multiple galaxies
    if test_theta is not None:
        if not uncertainty:
            # testing dataset
            _, k = load_galaxies(test_theta, "ndarray")
            x = np.split(test_x_raw, k, axis=0)

            return x[:num_entries]
        
        else:
            loaded_theta = load_h5(test_theta, "theta", "ndarray") if ".h5" in test_theta else load_csv(test_theta, "ndarray")
            
            if selection:
                loaded_theta, test_x_raw = np.column_stack(loaded_theta, np.array(test_x_raw[:, :2])), test_x_raw[:, 2:]
                
            _, unsorted_k = np.unique(loaded_theta[:, :8], axis=0, return_index=True)
            k = np.sort(unsorted_k)[1:]
            
            x = np.split(test_x_raw, k, axis=0)
            uncertainties = torch.tensor_split(torch.tensor(loaded_theta[:, 8:]), list(k), dim=0)

            
            return x[:num_entries], uncertainties[:num_entries]
            
        # reorder

    
    # Single galaxy
    else:
        if selection:
            positions, test_x_cooked = test_x_raw[:, :2], test_x_raw[:, 2:]
            return [test_x_cooked], positions
        else:
            test_x_cooked = [test_x_raw]
            return test_x_cooked



def prep_posterior(model:str|list[str], 
                   mcmc_settings:dict[str, str|dict[str,str|int]],
                   uncertainty: bool = False):
    """
    Load the model and prepare for MCMC (by setting settings)
    
    returns the posterior ready to be sampled
    
    params:
    - model: file location to the model
    - mcmc_settings: settings for MCMC
    - uncertainty_condition: Value of uncertainty that stays constant. None if no uncertainty in inference\
        use torch.nan to represent dimensions to sample and floats to represent the fixed values
        
    """
    
    inference = load_pickle(model)
    
    # inference._neural_net.to("cpu")
    if not uncertainty:    
        posterior = inference.build_posterior(**mcmc_settings)
    
    elif uncertainty:
        if isinstance(model, str):
            likelihood_estimator = inference._neural_net
        else:
            likelihood_estimator = CombinedLikelihoodEstimator(*model)
        
        
        prior = generate_prior(uncertainty= False, realistic_gamma=False)
        
        potential_fn, parameter_transform = likelihood_estimator_based_potential_with_uncertainty(
        likelihood_estimator, prior, x_o=None, uncertainties=None
        )
        
        
        posterior = MCMCPosteriorWithUncertainty(
        potential_fn, proposal=prior, 
        theta_transform=parameter_transform, 
        method = mcmc_settings["mcmc_method"],
        **mcmc_settings["mcmc_parameters"]
        )
    
    
    return posterior

def sample_single_galaxy(i, 
                         posterior, 
                         x_o: torch.Tensor,
                         uncertainty:torch.Tensor = None) -> torch.Tensor:
    """
    sample a single galaxy using MCMC
    
    params:
    - i: index of the galaxy, used for timing purposes
    - posterior; the posterior to sampel from
    - x_o: the data (stars) we want to run MCMC on
    - uncertainty: uncertainty for x_o
    """
    
    
    import warnings
    warnings.filterwarnings("ignore", message="An x with a batch size of")
    warnings.filterwarnings("ignore", message="As of sbi v0.19.0")
    
    
    start_time = time.perf_counter()
    print(f"starting {i}th galaxy")
    
    with torch.no_grad():
        if uncertainty is not None:
            samples = posterior.sample((6400,), x=x_o, uncertainty=uncertainty, show_progress_bars=True).numpy()
        else:
            samples = posterior.sample((6400,), x=x_o, show_progress_bars=True).numpy()
            

    
    end_time = time.perf_counter()
    print(f"Galaxy {i} took {end_time - start_time:.4f} seconds")
    
    # gc.collect()
    
    return samples


def run_mcmc(test_x: list[torch.Tensor],
             posterior, 
             n_galaxies_at_once:int,
             uncertainties: list[torch.Tensor] = None) -> list[np.ndarray]:
    """
    Run MCMC for all the galaxies
    
    returns a list of MCMC samples
    
    params:
        test_x: the test galaxies
        posterior: posterior that can be sampled from
        n_galaxy_at_once: run MCMC for this number of galaxies at once
        uncertainties: uncertainty for each star
    """
    
    print(f"Starting parallel MCMC for {len(test_x)} galaxies...")
    start_time = time.perf_counter()
    
    if uncertainties is None:
        final_samples = Parallel(n_jobs=n_galaxies_at_once, verbose=10)(
            delayed(sample_single_galaxy)(i, posterior, x_o) 
            for i, x_o in enumerate(test_x)
        )
    else:
        final_samples = Parallel(n_jobs=n_galaxies_at_once, verbose=10)(
            delayed(sample_single_galaxy)(i, posterior, test_x[i], uncertainty=uncertainties[i]) 
            for i in range(len(test_x))
        )
    
    end_time = time.perf_counter()
    print(f"Total time for {len(test_x)} galaxies: {end_time - start_time:.2f} seconds")
    
    return final_samples

def save_samples(samples:list[np.ndarray], 
                 file_path:str) -> None:
    """
    Saves the samples in a .pkl file for more than 1 galaxy, otherwise, in a csv file
    
    params:
    - samples: sampels to be saved
    - file_path: file location to be saved in 
    """

    if len(samples) == 1:
        save_csv(samples[0], file_path)
    else:
        save_pickle(samples, file_path)
    

if __name__ == "__main__":
    
    torch.set_num_threads(4)
    
    # MCMC settings - With Uncertainties
    mcmc_settings = {"mcmc_method":"slice_np_vectorized", 
                     "mcmc_parameters":{"warmup_steps":500,
                                    "num_chains": 32,
                                    "num_workers": 1,
                                    "init_strategy": "sir",
                                    "thin": 1}}
                                        
                                        
    # Example code for mass density
    prof = "cusp"
    # mock = "D"
    
    # print(mock)
    test_x = prep_data(f"./8d_theta/model_7_1/3d/mass_density_{prof}.csv",
                       train_x= "./8d_theta/model_8/5d/train_x.h5",
                       dim=3,)
    
    posterior = prep_posterior(f"./8d_theta/model_8/3d/inference.pkl",
                               mcmc_settings,
                               uncertainty=True)
    
    # uncertainty = load_csv(f"./8d_theta/model_8/mock/data/Mock{mock}_unc.csv", "Tensor")
    uncertainty = torch.full((100,1), 1)
    final_samples = run_mcmc(test_x, posterior, 1, 
                             uncertainties=[torch.log10(uncertainty)])
    
    save_samples(final_samples,
                 f"8d_theta/model_8/3d/mass_density_samples_{prof}_6400_t.csv")
    
# ======================================================================================================
    # # # MCMC settings - P(v| x, y, sigma, theta)
    # mcmc_settings = {"mcmc_method":"slice_np_vectorized", 
    #                  "mcmc_parameters":{"warmup_steps":500,
    #                                 "num_chains":16,
    #                                 "num_workers": 1,
    #                                 "init_strategy": "sir",
    #                                 "thin": 1}}
                                        
                                        
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
    
    # posterior = prep_posterior(f"./8d_theta/model_9/{dim}d/inference.pkl",
    #                            mcmc_settings,
    #                            uncertainty=True)
        
    # final_samples = run_mcmc(test_x, posterior, 1, 
    #                          uncertainties=[torch.column_stack((uncertainty, position))])
    
    # save_samples(final_samples,
    #              f"8d_theta/model_9/{dim}d/mock/Mock{mock}_samples.csv")
    
# ======================================================================================================
    # # MCMC settings - No Uncertainties
    # mcmc_settings = {"mcmc_method":"slice_np_vectorized", 
    #                  "mcmc_parameters":{"warmup_steps":500,
    #                                 "num_chains":16,
    #                                 "num_workers": 1,
    #                                 "init_strategy": "sir",
    #                                 "thin": 1}}
                                        
                                        
    # # Example code for mass density
    # prof = "core"
    
    # print(prof)
    # test_x = prep_data(f"./8d_theta/model_5_2/5d/mass_density_{prof}.csv",
    #                    train_x= "./8d_theta/model_5_2/5d/train_x.csv",)
    
    # posterior = prep_posterior(f"./8d_theta/model_5_2/5d/inference.pkl",
    #                            mcmc_settings,
    #                            uncertainty=False)
        
    # final_samples = run_mcmc(test_x, posterior, 1, 
    #                          uncertainties=None)
    
    # save_samples(final_samples,
    #              f"8d_theta/model_5_2/5d/mass_density_samples_{prof}.csv")
    
# ======================================================================================================
    # # Example code for mass density (SNLE)
        
    # test_x_core = prep_data(f"./8d_theta/model_1/mass_density_core.csv",
    #         train_x= "./8d_theta/model_3/train_x.csv")
    # test_x_cusp = prep_data(f"./8d_theta/model_1/mass_density_cusp.csv",
    #         train_x= "./8d_theta/model_3/train_x.csv")
    
    # def parallel(prof):
    #     posterior = prep_posterior(f"./8d_theta/model_4/inference_{prof}/inference_r{i}.pkl",
    #                             mcmc_settings)
    #     if prof == "core":
    #         test_x = test_x_core
    #     elif prof == "cusp":
    #         test_x = test_x_cusp
            
    #     final_samples = run_mcmc(test_x, posterior, 1)
        
    #     save_samples(final_samples,
    #                 f"8d_theta/model_4/inferernce_{prof}/mass_density_samples_r{i}.csv")
        
    # for i in range(5):
    #     Parallel(n_jobs=2, verbose=10)(delayed(parallel)(prof) for prof in ["core", "cusp"])
        
# ======================================================================================================

    # # Example code for normal evaluation
    # mcmc_settings = {"mcmc_method":"slice_np_vectorized", 
    #                 "mcmc_parameters":{"warmup_steps":500,
    #                             "num_chains":16,
    #                             "num_workers": 1,
    #                             "init_strategy": "sir",
    #                             "thin": 1}}
    
    # test_x, uncertainties = prep_data("./8d_theta/model_8/3d/test_x.h5",
    #                    test_theta="./8d_theta/model_8/3d/test_theta.h5",
    #                    train_x= "./8d_theta/model_8/3d/train_x.h5",
    #                    num_entries=2000,
    #                    uncertainty=True)

    # posterior = prep_posterior("./8d_theta/model_8/3d/inference.pkl",
    #                            mcmc_settings,
    #                            uncertainty=True)
    
    # final_samples = run_mcmc(test_x, posterior, 8, uncertainties=uncertainties)
    
    # save_samples(final_samples,
    #              "8d_theta/model_8/3d/samples.pkl")