import agama
import torch 
import numpy as np
from astropy import units as u

from sbi.inference import SNLE

import pandas as pd
import pickle

from python_scripts.prior_generation import generate_prior

torch.set_num_threads(4)

# set agama unit to be in Msun, kpc, km/s
agama.setUnits(mass=1 * u.Msun, length=1*u.kpc, velocity=1 * u.km /u.s)
agama.setRandomSeed(13)
torch.manual_seed(13)
np.random.seed(13)

dataset = "poisson"

theta_train = torch.from_numpy(np.array(pd.read_csv(f"./data/training_theta({dataset}).csv", header=None))).float()
x_train = torch.from_numpy(np.array(pd.read_csv(f"./data/training_x({dataset}).csv", header=None))).float()

prior_sbi = generate_prior()
inference = SNLE(prior=prior_sbi)
inference.append_simulations(theta_train, x_train)
arg = {
        "training_batch_size": 512,
        "learning_rate": 0.001347,
        "validation_fraction": 0.1,
        "stop_after_epochs": 10,
        "max_num_epochs": 2^31 - 1,
        "clip_max_norm": 5.0,
        "resume_training": False,
        "discard_prior_samples": False,
        "retrain_from_scratch": False,
        "show_train_summary": True,
        "dataloader_kwargs": None
}

likelihood_estimator = inference.train(**arg)

with open(f"./model/likelihood_estimator({dataset}).pkl", "wb") as handle:
    pickle.dump(likelihood_estimator, handle)
    
with open(f"./model/inference({dataset}).pkl", "wb") as handle:
    pickle.dump(inference, handle)