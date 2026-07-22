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

from sbi.inference import SNLE
from sbi.utils import likelihood_nn
import pandas as pd
import pickle
from standardization import standardize
from prior_generation import generate_prior
from object_handler import save_pickle, load_csv, load_galaxies, load_h5
import time
from datetime import datetime
torch.set_num_threads(4)

# set agama unit to be in Msun, kpc, km/s
agama.setUnits(mass=1 * u.Msun, length=1*u.kpc, velocity=1 * u.km /u.s)
agama.setRandomSeed(13)
torch.manual_seed(13)
np.random.seed(13)


def prep_data(train_theta:str,
              train_x:str,
              standardization: bool = True,
              uncertainty:bool = False) -> tuple[torch.Tensor]:
    """
    Prepare the data for training
    
    params:
    - train_theta: file path to training theta
    - train_x: file path to training x
    - standardization: if standardization is needed
    - uncertainty: to include uncertainty in the inference or not; use to determine file format
    """

    if not uncertainty:
        theta, k = load_galaxies(train_theta, "Tensor")
        train_theta = torch.repeat_interleave(theta, k, dim=0)
    else:
        # train_theta = load_csv(train_theta, "Tensor")
        train_theta = load_h5(train_theta, "theta", "Tensor") if ".h5" in train_theta else load_csv(train_theta, "Tensor")
    
    # train_x_raw = load_csv(train_x, "Tensor")
    train_x_raw = load_h5(train_x, "x", "Tensor") if ".h5" in train_x else load_csv(train_x, "Tensor")

    train_x = standardize(train_x_raw)[0] if standardization else train_x_raw
    
    return train_theta, train_x


def prep_inference(train_theta: torch.Tensor,
                   train_x: torch.Tensor,
                   likelihood_estimator_settings: dict[str, str | int] | None = None, 
                   uncertainty:bool = False
                   ) -> SNLE:
    """
    Prepare the inference for training
    
    params:
    - train_theta: The tensor of trarining thetas
    - train_x: The tensor of trarining x's
    - likelihood_settings: the settings for customized structures. None if default settings 
    - uncertainty: to include uncertainty in the inference or not; used for generating the prior
    """

    prior_sbi = generate_prior(uncertainty=uncertainty)

    if likelihood_estimator_settings is not None:
        density_estimator =likelihood_nn(**likelihood_estimator_settings)
        inference = SNLE(prior=prior_sbi, density_estimator=density_estimator)
    else:
        inference = SNLE(prior=prior_sbi)
    
    inference.append_simulations(train_theta, train_x)
    
    return inference

def train_model(inference:SNLE,
                training_settings: dict[str, int | bool]) -> SNLE:
    
    """
    Trains the model
    
    params:
    - inference: the model
    - training_settings: training settings
    """
    start_time = time.perf_counter()
    
    print(f"{datetime.now()}: Beginning model training")
    
    inference.train(**training_settings)
    
    end_time = time.perf_counter()
    print(f"Training took {end_time - start_time}")
    
    return inference


if __name__ == "__main__":
    train_theta, train_x = prep_data("./8d_theta/model_8/5d/train_theta.h5",
                                     "./8d_theta/model_8/5d/train_x.h5",
                                     uncertainty=True)

    ### For P(v| x, y, sigma, theta) ###
    train_theta = torch.column_stack((train_theta, train_x[:, :2]))
    train_x = train_x[:, 2:]

    likelihood_estimator_settings = {'model': 'maf',
                                    'hidden_features': 75,
                                    'num_transforms': 9,
                                    'num_bins': 11}
    
    inference = prep_inference(train_theta,
                               train_x,
                               likelihood_estimator_settings=likelihood_estimator_settings,
                               uncertainty=True)

    arg = {
            "training_batch_size": 4096,
            "learning_rate": 0.00027967246181336485,
            "validation_fraction": 0.1,
            "stop_after_epochs": 20,
            "max_num_epochs": 100,
            "clip_max_norm": 5.0,
            "resume_training": False,
            "discard_prior_samples": False,
            "retrain_from_scratch": False,
            "show_train_summary": True,
            # "dataloader_kwargs": {"num_workers": 2, 
            #                         "persistent_workers": True}
    }
    
    
    inference = train_model(inference, arg)
    
    save_pickle(inference, "./8d_theta/model_9/5d/inference.pkl")

