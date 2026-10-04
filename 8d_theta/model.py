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

from torch.optim.lr_scheduler import CosineAnnealingLR
from sbi.inference import SNLE
from sbi.utils import likelihood_nn
import pandas as pd
import pickle
from standardization import standardize
from prior_generation import generate_prior
from object_handler import save_pickle, load_csv, load_galaxies, load_h5
import time
from datetime import datetime
from copy import deepcopy
torch.set_num_threads(4)

# set agama unit to be in Msun, kpc, km/s
agama.setUnits(mass=1 * u.Msun, length=1*u.kpc, velocity=1 * u.km /u.s)
agama.setRandomSeed(13)
torch.manual_seed(13)
np.random.seed(13)

import os, resource, psutil

def mem(tag=""):
    cur = psutil.Process(os.getpid()).memory_info().rss / 1e9
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6   # KB → GB on Linux
    print(f"{tag} current: {cur:.2f} GB | peak so far: {peak:.2f} GB", flush=True)


import torch.utils.data

# Monkey-patch SubsetRandomSampler to prevent std::bad_alloc on large datasets
def patched_subset_random_sampler_iter(self):
    for i in torch.randperm(len(self.indices), generator=self.generator).tolist():
        yield self.indices[i]


torch.utils.data.SubsetRandomSampler.__iter__ = patched_subset_random_sampler_iter

def prep_data(train_theta:str,
              train_x:str,
              standardization: bool = True,
              uncertainty:bool = False,
              dim=5,
              cut:int = None) -> tuple[torch.Tensor]:
    """
    Prepare the data for training
    
    params:
    - train_theta: file path to training theta
    - train_x: file path to training x
    - standardization: if standardization is needed
    - uncertainty: to include uncertainty in the inference or not; use to determine file format
    - dim: dimension of stellar kinematics
    - cut: cut the data set for debugging purposes
    """

    # Load x and theta (5d)
    
    if not uncertainty:
        # If no uncertainty in inference
        theta, k = load_galaxies(train_theta, "Tensor")
        prepped_theta = torch.repeat_interleave(theta, k, dim=0)
    else:
        # If uncertainty in inference
        prepped_theta = load_h5(train_theta, "theta", "Tensor") if ".h5" in train_theta else load_csv(train_theta, "Tensor")
    
    train_x_raw = load_h5(train_x, "x", "Tensor") if ".h5" in train_x else load_csv(train_x, "Tensor")

    # Reduce to the desired dimension
    if dim != train_x_raw.shape[1]:
        if dim == 4:
            train_x_raw = train_x_raw[:, :4]
            prepped_theta = prepped_theta[:, :10] if uncertainty else prepped_theta
        elif dim == 3:
            train_x_raw = train_x_raw[:, (0, 1, 4)]
            prepped_theta = prepped_theta[:, (0, 1, 2, 3, 4, 5, 6, 7, 10)] if uncertainty else prepped_theta

    if cut is not None:
        prepped_theta = prepped_theta[:cut]
        train_x_raw = train_x_raw[:cut]

    # Standardize the x
    prepped_x = standardize(train_x_raw)[0] if standardization else train_x_raw    
    return prepped_theta, prepped_x


def prep_inference(train_theta: torch.Tensor,
                   train_x: torch.Tensor,
                   likelihood_estimator_settings: dict[str, str | int] | None = None, 
                #    uncertainty:bool = False
                   ) -> SNLE:
    """
    Prepare the inference for training
    
    params:
    - train_theta: The tensor of trarining thetas
    - train_x: The tensor of trarining x's
    - likelihood_settings: the settings for customized structures. None if default settings 
    - uncertainty: to include uncertainty in the inference or not; used for generating the prior
    """

    # prior_sbi = generate_prior(uncertainty=uncertainty)

    if likelihood_estimator_settings is not None:
        density_estimator =likelihood_nn(**likelihood_estimator_settings)
        # inference = SNLE(prior=prior_sbi, density_estimator=density_estimator)
        inference = SNLE(density_estimator=density_estimator)
    else:
        # inference = SNLE(prior=prior_sbi)
        inference = SNLE()
    
    
    inference.append_simulations(train_theta, train_x)
    
    return inference

def train_model(inference:SNLE,
                training_settings: dict[str, int | bool],
                scheduler: bool = False) -> SNLE:
    
    """
    Trains the model
    
    params:
    - inference: the model
    - scheduler: apply scheduler to the learning rate
    - training_settings: training settings
    """
    start_time = time.perf_counter()
    
    print(f"{datetime.now()}: Beginning model training")
    
    if not scheduler:
        inference.train(**training_settings)
    else:
        settings = training_settings.copy()
        epochs = settings["max_num_epochs"]
        patience = settings["stop_after_epochs"]
        
        settings["max_num_epochs"] = 1
        settings["stop_after_epochs"] = epochs + 1
        # first epoch
        inference.train(**settings)
        
        settings["resume_training"] = True
        sched = CosineAnnealingLR(inference.optimizer, T_max=epochs)
        best_val = -float("inf")
        best_net = deepcopy(inference._neural_net)
        best_epoch = 0

        for i in range(1, epochs):
            settings["max_num_epochs"] += 1
            inference.train(**settings)
            sched.step()

            val = inference._summary["validation_log_probs"][-1]
            if val > best_val + 0.01:
                best_val, best_epoch = val, i
                best_net = deepcopy(inference._neural_net)
            if i - best_epoch >= patience:
                print(f"stopping at epoch {i}; best {best_val:.4f} at epoch {best_epoch}")
                break
            
        inference._neural_net = best_net
        inference._best_val_log_prob = best_val
        print(f"best validation log-prob {best_val:.4f} at epoch {best_epoch}/{i}")
    
    end_time = time.perf_counter()
    print(f"Training took {end_time - start_time}")
    
    return inference


if __name__ == "__main__":
    train_theta, train_x = prep_data("./8d_theta/model_14/3d/train_theta.h5",
                                     "./8d_theta/model_14/3d/train_x.h5",
                                     uncertainty=True,
                                     dim=3,)
                                    #  cut=100000)
    mem("start")

    # ### For P(v| x, y, sigma, theta) ###
    # train_theta = torch.column_stack((train_theta, train_x[:, :2]))
    # train_x = train_x[:, 2:]


    likelihood_estimator_settings = {'model': 'maf',
                                    'hidden_features': 71,
                                    'num_transforms': 8,
                                    'num_bins': 12}
    
    inference = prep_inference(train_theta,
                               train_x,
                               likelihood_estimator_settings=likelihood_estimator_settings,)
    mem("after prep_data")
    arg = {
            "training_batch_size": 8192,
            "learning_rate": 0.0006921732022808391,
            "validation_fraction": 0.1,
            "stop_after_epochs": 20,
            "max_num_epochs": 100,
            "clip_max_norm": 3.0,
            "resume_training": False,
            "discard_prior_samples": False,
            "retrain_from_scratch": False,
            "show_train_summary": False,
            # "dataloader_kwargs": {"num_workers": 2, 
            #                         "persistent_workers": True}
    }
    
    
    inference = train_model(inference, arg, scheduler=True)
    
    save_pickle(inference, "./8d_theta/model_16/3d/inference.pkl", override=False)

