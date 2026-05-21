import agama
import torch 
import numpy as np
from astropy import units as u

from sbi.inference import SNLE
from sbi.utils import likelihood_nn
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



torch.set_num_threads(4)
prior_sbi = generate_prior()

density_estimator =likelihood_nn(model="nsf", 
                                 hidden_features = 96,
                                 num_transforms = 8,
                                 num_bins = 7)

inference = SNLE(prior=prior_sbi, density_estimator=density_estimator)
inference.append_simulations(theta_train, x_train)
arg = {
        "training_batch_size": 1024,
        "learning_rate": 0.0048266,
        "validation_fraction": 0.1,
        "stop_after_epochs": 10,
        "max_num_epochs": 2^31 - 1,
        "clip_max_norm": 5.0,
        "resume_training": False,
        "discard_prior_samples": False,
        "retrain_from_scratch": False,
        "show_train_summary": True,
        "dataloader_kwargs": {"num_workers": 2, 
                                "persistent_workers": True}
}

likelihood_estimator = inference.train(**arg)

# lr: 0.001347, tbs: 512

with open(f"./model/likelihood_estimator({dataset}_d_20k).pkl", "wb") as handle:
    pickle.dump(likelihood_estimator, handle)
    
with open(f"./model/inference({dataset}_d_20k).pkl", "wb") as handle:
    pickle.dump(inference, handle)