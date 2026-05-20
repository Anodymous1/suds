import agama
import torch 
import numpy as np
from astropy import units as u

from sbi.utils import BoxUniform
from sbi.inference import SNLE, simulate_for_sbi, prepare_for_sbi

from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
import pickle

from galaxy_generation import generate_galaxy_multiple
from prior_generation import generate_prior

import optuna
import gc

torch.set_num_threads(4)

theta_train = torch.from_numpy(np.array(pd.read_csv("./data/training_theta(100).csv", header=None))).float()
x_train = torch.from_numpy(np.array(pd.read_csv("./data/training_x(100).csv", header=None))).float() 

def objective(trial):
        
    global theta_train
    global x_train
    
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    training_batch_size = trial.suggest_categorical("training_batch_size", [128, 256, 512, 1024])
    
    
    prior_sbi = generate_prior()
    inference = SNLE(prior=prior_sbi)
    inference.append_simulations(theta_train, x_train)
    
    likelihood_estimator = inference.train(learning_rate = learning_rate,
                                           training_batch_size = training_batch_size,
                                           stop_after_epochs = 10,
                                           dataloader_kwargs= {"num_workers": 4, 
                                                                "persistent_workers": True}
    )
    
    val = inference._summary["best_validation_log_prob"][0]
    
    
    del inference
    del likelihood_estimator
    gc.collect()
    
    return val


agama.setRandomSeed(13)
torch.manual_seed(13)
np.random.seed(13)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

import pickle
with open("./model/tune.pkl", "wb") as handle:
    pickle.dump(study, handle)
    
print(study.best_params)
