from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"     # agama
os.environ["MKL_NUM_THREADS"] = "1"     # numpy, scipy
os.environ["OPENBLAS_NUM_THREADS"] = "1"    # numpy
os.environ["NUMEXPR_NUM_THREADS"] = "1"     # pandas
import random
import agama
import torch 
import numpy as np

import pandas as pd
import pickle
from galaxy_generation import generate_galaxy_multiple
import time
from sbi.inference import SNLE
from standardization import standardize
from object_handler import save_pickle, load_pickle, load_csv
# from sbi.inference.posteriors.mcmc_posterior import MCMCPosterior
from sbi.utils import likelihood_nn

agama.setRandomSeed(13)
torch.manual_seed(13)
np.random.seed(13)
random.seed(13)

def initialize_training(file_path_model: str,
                        x_o_path:str,
                        likelihood_estimator_settings: dict[str, str|int],
                        mcmc_settings: dict[str, str|dict[str,str|int]],
                        file_path_standardize: str|None) -> tuple[SNLE, int|None]:
    """
    Initialize the training
    
    returns the original model, initial proposal, mean and std of training dataset, the galaxy we want to specialize
    
    params:
    - file_path_model: file directory to the original mdoel
    - x_o_path: file path to x_o
    - likelihood_estimator_settings: settings for likelihood_nn
    - mcmc_settings: mcmc settings
    - file_path_standardize: training data set for standardizing
    """
    
    
    x_o = load_csv(x_o_path, "Tensor")
    
    inference = load_pickle(file_path_model)
    inference._build_posterior = likelihood_nn(**likelihood_estimator_settings)
    proposal = generate_proposal(inference, mcmc_settings, x_o)
        
    
    if file_path_standardize is not None:
        x = standardize(load_csv(file_path_standardize, "Tensor"))
        return inference, proposal, x[1], x[2], x_o
    
    else:
        return inference, proposal, None, None, x_o
    
def generate_new_data(proposal, 
                      num_samples: int,
                      std_mean: torch.Tensor,
                      std_std: torch.Tensor) -> tuple[torch.Tensor]:
    """
    Generate new data to be added to the model to retrain
    
    params:
    - proposal: the proposal
    - num_samples: number of MCMC samples to be generated
    - std_mean: standardization mean
    - std_std: standardization standard deviation
    """
    
    # sample posterior as prior
    theta = proposal.sample((num_samples,))
    
    # simualte new x
    n_stars = np.random.poisson(100, size=num_samples)
    new_x = generate_galaxy_multiple(theta, n_stars, 4)
    
    if (std_mean is not None) or (std_std is not None):
        new_x = (new_x - std_mean) / std_std
    
    new_theta = torch.repeat_interleave(theta, n_stars, dim=0)
    
    return new_theta, new_x
 
 
def generate_proposal(inference: SNLE, 
                      mcmc_settings: dict[str, str|dict[str,str|int]],
                      x_o: torch.Tensor):
    """
    Generate the proposal (posterior)
    
    params:
    - inference: the model
    - mcmc_settings: MCMC settings
    - x_o: the galaxy we want to specialize
    """
    
             
    # make posterior
    posterior = inference.build_posterior(**mcmc_settings)
    proposal = posterior.set_default_x(x_o)
    
    return proposal
 
 
def train_model(file_path_model: str,
                x_o_path: str,
                file_path_save: str,
                train_settings: dict[str, float|bool],
                mcmc_settings: dict[str, str|dict[str,str|int]],
                likelihood_estimator_settings: dict[str, str|int],
                num_samples: int = 1000,
                num_rounds: int = 5,
                file_path_standardize: str|None = None):
    """
    train the model
    
    params:
    - file_path_model: file directory to the original model
    - x_o_path: file path to x_o
    - file_path_save: where to save the models
    - train_settings: training settings (eg batch size, learning rate etc)
    - mcmc_settings: MCMC settings
    - likelihood_estimator_settings: settings for likelihood_nn
    - num_samples: number of MCMC samples to generate per round
    - num_rounds: number of rounds
    - file_path_standardize: the file directory of the training dataset used to standardize x
    """
    
    
    inference, proposal, std_mean, std_std, x_o = initialize_training(file_path_model,
                                                                      x_o_path,
                                                                      likelihood_estimator_settings,
                                                                      mcmc_settings,
                                                                      file_path_standardize=file_path_standardize)

    # Begin training
    start_time_whole = time.perf_counter()

    for i in range(num_rounds):
        print(f"Beginning round {i}")
        
        start_time = time.perf_counter()
        new_theta, new_x = generate_new_data(proposal,
                                             num_samples,
                                             std_mean,
                                             std_std)
        # retrain model
        inference.append_simulations(new_theta, new_x).train(**train_settings)
        
        # new proposal
        proposal = generate_proposal(inference, mcmc_settings, x_o)
        
        # save model
        save_pickle(inference, f"{file_path_save}/inference_r{i}.pkl")
        
        end_time = time.perf_counter()
        print(f"Round {i} took {end_time - start_time:.4f} seconds")

    end_time_whole = time.perf_counter()
    print(f"Entire training took {end_time_whole - start_time_whole:.4f} seconds")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message="An x with a batch size of")
    warnings.filterwarnings("ignore", message="As of sbi v0.19.0")
    
    likelihood_estimator_settings = {"model":"nsf", 
                                "hidden_features": 128,
                                "num_transforms": 8,
                                "num_bins": 8}
    train_settings = {
        "training_batch_size": 1024,
        "learning_rate":0.0017034433770023658,
        "validation_fraction": 0.1,
        "stop_after_epochs": 20,
        "max_num_epochs": 2 ** 31 - 1,
        "clip_max_norm": 5.0,
        "resume_training": False,
        "discard_prior_samples": False,
        "retrain_from_scratch": False,
        "show_train_summary": True,
    }
    mcmc_settings = {"mcmc_method":"slice_np_vectorized", 
                    "mcmc_parameters":{"warmup_steps":500,
                                "num_chains":16,
                                "num_workers": 1,
                                "init_strategy": "sir",
                                "thin": 1}}
    
    prof = "cusp"
    
    train_model("./8d_theta/model_1/inference.pkl",
                f"./8d_theta/model_4/x_o_{prof}.csv",
                f"./8d_theta/model_4/inference_{prof}",
                train_settings,
                mcmc_settings,
                likelihood_estimator_settings,
                num_samples=1000,
                num_rounds=5,
                file_path_standardize="./8d_theta/model_1/train_x.csv")