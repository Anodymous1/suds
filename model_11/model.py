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
from standardization import standardize, get_standard
from prior_generation import generate_prior


torch.set_num_threads(4)

# set agama unit to be in Msun, kpc, km/s
agama.setUnits(mass=1 * u.Msun, length=1*u.kpc, velocity=1 * u.km /u.s)
agama.setRandomSeed(13)
torch.manual_seed(13)
np.random.seed(13)

# dataset = "poisson"

theta_train = np.array(pd.read_csv(f"./model_11/train_theta.csv", header=None))
x_train = np.array(pd.read_csv(f"./model_11/train_x.csv", header=None))

t, x = standardize(theta_train, x_train)
x_train = x[0]
# x_train = torch.tensor((x_train - x[1]) / x[2]).float()
x_train = torch.tensor(x_train).float()

theta_train = torch.tensor(theta_train).float()
torch.set_num_threads(4)
prior_sbi = generate_prior()

density_estimator =likelihood_nn(model="nsf", 
                                 hidden_features = 128,
                                 num_transforms = 7,
                                 num_bins= 8)

inference = SNLE(prior=prior_sbi, density_estimator=density_estimator)
# inference = SNLE(prior=prior_sbi)
inference.append_simulations(theta_train, x_train)
arg = {
        "training_batch_size": 1024,
        "learning_rate": 0.001113340140740274,
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

print("now start training")
likelihood_estimator = inference.train(**arg)

# lr: 0.001347, tbs: 512

# with open(f"./model 6/likelihood_estimator_model_8.pkl", "wb") as handle:
#     pickle.dump(likelihood_estimator, handle)
    
with open(f"./model_11/inference_model_11_test.pkl", "wb") as handle:
    pickle.dump(inference, handle)