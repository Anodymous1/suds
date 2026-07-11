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
from object_handler import save_pickle, load_csv, load_galaxies


torch.set_num_threads(4)

# set agama unit to be in Msun, kpc, km/s
agama.setUnits(mass=1 * u.Msun, length=1*u.kpc, velocity=1 * u.km /u.s)
agama.setRandomSeed(13)
torch.manual_seed(13)
np.random.seed(13)


def prep_data(train_theta:str,
              train_x:str,
              standardization: bool = True) -> tuple[torch.Tensor]:
    """
    Prepare the data for training
    
    params:
    - train_theta: file path to training theta
    - train_x: file path to training x
    - standardization: if standardization is needed
    """

    theta, k = load_galaxies(train_theta, "Tensor")
    train_theta = torch.repeat_interleave(theta, k, dim=0)
    
    train_x_raw = load_csv(train_x, "Tensor")

    train_x = standardize(train_x_raw)[0] if standardization else train_x_raw
    
    return train_theta, train_x


def prep_inference(train_theta: torch.Tensor,
                   train_x: torch.Tensor,
                   likelihood_estimator_settings: dict[str, str | int] | None = None, 
                   ) -> SNLE:
    """
    Prepare the inference for training
    
    params:
    - train_theta: The tensor of trarining thetas
    - train_x: The tensor of trarining x's
    - likelihood_settings: the settings for customized structures. None if default settings 
    """

    prior_sbi = generate_prior()

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
    
    inference.train(**training_settings)
    
    return inference


if __name__ == "__main__":
    train_theta, train_x = prep_data("./8d_theta/model_5/train_theta.csv",
                                     "./8d_theta/model_5/train_x.csv")


    likelihood_estimator_settings = {"model":"nsf", 
                                    "hidden_features": 128,
                                    "num_transforms": 8,
                                    "num_bins": 8}
    
    inference = prep_inference(train_theta,
                               train_x,
                               likelihood_estimator_settings=likelihood_estimator_settings)

    arg = {
            "training_batch_size": 2048,
            "learning_rate": 0.0017034433770023658,
            "validation_fraction": 0.1,
            "stop_after_epochs": 20,
            "max_num_epochs": 2 ** 31 - 1,
            "clip_max_norm": 5.0,
            "resume_training": False,
            "discard_prior_samples": False,
            "retrain_from_scratch": False,
            "show_train_summary": True,
            # "dataloader_kwargs": {"num_workers": 2, 
            #                         "persistent_workers": True}
    }
    
    inference = train_model(inference, arg)
    
    save_pickle(inference, "./8d_theta/model_5/inference_2048.pkl")

