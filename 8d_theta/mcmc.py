from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"



import agama
import torch 
import numpy as np
from astropy import units as u
import pandas as pd
import pickle
import time
from standardization import standardize
from object_handler import save_pickle, save_csv, load_csv
from joblib import Parallel, delayed

# set agama unit to be in Msun, kpc, km/s
agama.setUnits(mass=1 * u.Msun, length=1*u.kpc, velocity=1 * u.km /u.s)

agama.setRandomSeed(13)
torch.manual_seed(13)
np.random.seed(13)


def prep_data(test_x: str,
              test_theta: str|None = None,
              train_x: str|None = None,
              num_entries:int = 100) -> list[torch.Tensor]:
    """
    Prepare the data for MCMC
    
    params:
    - test_x: file location for test data of x
    - test_theta: file location for test data of theta. None if MCMC is ran on 1 galaxy
    - train_x: file location of the training x. None if standardization is not needed
    - num_entries: only keep the first num_entries test data
    """
    
    test_x_raw = load_csv(test_x, "Tensor")
    
    # standardization
    if train_x is not None:
        train_x = load_csv(train_x, "Tensor")
        x = standardize(train_x)
        test_x_raw = (test_x_raw - x[1])/ x[2]
    
    
    # Non-single galaxy
    if test_theta is not None:
        # testing dataset
        test_theta_raw = load_csv(test_theta, "ndarray")

        # reorder
        _, index = np.unique(test_theta_raw, axis=0, return_index=True)
        index = np.sort(index)
        test_x = np.split(test_x_raw, index, axis=0)[1:]

        return test_x[:num_entries]
    
    # Single galaxy
    else:
        test_x = [test_x_raw]
        
        return test_x


def prep_posterior(model:str, 
                   mcmc_settings:dict[str, str|dict[str,str|int]]):
    """
    Load the model and prepare for MCMC (by setting settings)
    
    returns the posterior ready to be sampled
    
    params:
    - model: file location to the model
    """
    
    with open(model, 'rb') as file:
        # Load the object from the file
        inference = pickle.load(file)
    
    inference._neural_net.to("cpu")
    
    posterior = inference.build_posterior(**mcmc_settings)
    
    return posterior

def sample_single_galaxy(i, 
                         posterior, 
                         x_o: torch.Tensor) -> torch.Tensor:
    """
    sample a single galaxy using MCMC
    
    params:
    - i: index of the galaxy, used for timing purposes
    - posterior; the posterior to sampel from
    - x_o: the data (stars) we want to run MCMC on
    """
    
    
    import warnings
    warnings.filterwarnings("ignore", message="An x with a batch size of")
    warnings.filterwarnings("ignore", message="As of sbi v0.19.0")
    
    
    start_time = time.perf_counter()
    print(f"starting {i}th galaxy")
    
    with torch.no_grad():
        samples = posterior.sample((2000,), x=x_o, show_progress_bars=True).numpy()
    
    end_time = time.perf_counter()
    print(f"Galaxy {i} took {end_time - start_time:.4f} seconds")
    
    # gc.collect()
    
    return samples


def run_mcmc(test_x: list[torch.Tensor],
             posterior, 
             n_galaxies_at_once:int) -> list[np.ndarray]:
    """
    Run MCMC for all the galaxies
    
    returns a list of MCMC samples
    
    params:
        posterior: posterior that can be sampled from
        n_galaxy_at_once: run MCMC for this number of galaxies at once
    """
    
    print(f"Starting parallel MCMC for {len(test_x)} galaxies...")
    start_time = time.perf_counter()
    
    final_samples = Parallel(n_jobs=n_galaxies_at_once, verbose=10)(
        delayed(sample_single_galaxy)(i, posterior, x_o) 
        for i, x_o in enumerate(test_x)
    )
    
    # final_samples = []
    # for i, x_o in enumerate(test_x):
    #     samples = sample_single_galaxy(i, posterior, x_o)
    #     final_samples.append(samples)
    
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
    
    # MCMC settings
    mcmc_settings = {"mcmc_method":"slice_np_vectorized", 
                     "mcmc_parameters":{"warmup_steps":500,
                                    "num_chains":8,
                                    "num_workers": 1,
                                    "init_strategy": "sir",
                                    "thin": 1}}
                                        
                                        
    # # Example code for mass density
    # prof = "core"
    
    # test_x = prep_data(f"./8d_theta/model_1/mass_density_{prof}.csv",
    #                    train_x= "./8d_theta/model_1/train_x.csv")
    
    # posterior = prep_posterior(f"./8d_theta/model_1/inference.pkl",
    #                            mcmc_settings)
        
    # final_samples = run_mcmc(test_x, posterior, 1)
    
    # save_samples(final_samples,
    #              f"8d_theta/model_1/mass_density_samples_{prof}.csv")



    # Example code for normal evaluation
    test_x = prep_data("./8d_theta/model_1/test_x.csv",
                       test_theta="./8d_theta/model_1/test_theta.csv",
                       train_x= "./8d_theta/model_1/train_x.csv")

    posterior = prep_posterior("./8d_theta/model_1/inference.pkl",
                               mcmc_settings)
    final_samples = run_mcmc(test_x, posterior, 3)
    
    save_samples(final_samples,
                 "8d_theta/model_1/samples.pkl")