import agama
import torch 
import numpy as np
from astropy import units as u

from sbi.utils import BoxUniform
from sbi.inference import SNLE, simulate_for_sbi, prepare_for_sbi
from sbi.utils import likelihood_nn

from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
import pickle

from galaxy_generation import generate_galaxy_multiple
from prior_generation import generate_prior

import optuna
import gc


theta_train = torch.from_numpy(np.array(pd.read_csv("./data/training_theta(poisson).csv", header=None))).float()[:20088] 
x_train = torch.from_numpy(np.array(pd.read_csv("./data/training_x(poisson).csv", header=None))).float()[:20088] 




def objective(trial):
        
    torch.set_num_threads(4)
    
    global theta_train
    global x_train
    
    # Learning
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    training_batch_size = trial.suggest_categorical("training_batch_size", [512, 1024, 2048])
    
    
    # Normalizing flow
    hidden_features = trial.suggest_int("hidden_features", 64, 128)
    num_transforms = trial.suggest_int("num_transforms", 3, 10)
    num_bins = trial.suggest_int("num_bins", 5, 15)
    
    density_estimator =likelihood_nn(model="nsf", 
                                 hidden_features = hidden_features,
                                 num_transforms = num_transforms,
                                 num_bins = num_bins)

    
    prior_sbi = generate_prior()
    inference = SNLE(prior=prior_sbi, density_estimator=density_estimator)
    inference.append_simulations(theta_train, x_train)

    
    likelihood_estimator = inference.train(learning_rate = learning_rate,
                                           training_batch_size = training_batch_size,
                                           stop_after_epochs = 10,
                                           dataloader_kwargs= {"num_workers": 0}
    )

    
    
    val = inference._summary["best_validation_log_prob"][0]
    
    
    del inference
    del likelihood_estimator
    del density_estimator
    gc.collect()
    
    return val


agama.setRandomSeed(13)
torch.manual_seed(13)
np.random.seed(13)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

import pickle
with open("./model/tune(poisson_d_20k).pkl", "wb") as handle:
    pickle.dump(study, handle)
    
print(study.best_params)
