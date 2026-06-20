import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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
from standardization import standardize

import optuna
import gc


theta_train = np.array(pd.read_csv("./model_11/training_theta_model_14.csv", header=None))
x_train = np.array(pd.read_csv("./model_11/training_x_model_14.csv", header=None))

x_train = standardize(theta_train, x_train)[1][0]

theta_train, x_train = torch.tensor(theta_train[:199983]).float(), torch.tensor(x_train[:199983]).float()
# theta_train, x_train = torch.tensor(theta_train).float(), torch.tensor(x_train).float()

def objective(trial):
        
    torch.set_num_threads(4)

    global theta_train
    global x_train
    
    # Learning
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    # training_batch_size = trial.suggest_categorical("training_batch_size", [256, 512, 1024, 2048, 4096])
    training_batch_size = 512
    
    
    # Normalizing flow
    model = trial.suggest_categorical("model", ["maf", "nsf"])
    hidden_features = trial.suggest_int("hidden_features", 32, 128)
    num_transforms = trial.suggest_int("num_transforms", 3, 8)
    num_bins = trial.suggest_int("num_bins", 4, 12)
    # patience = trial.suggest_int("patience", 10, 30)
    patience = 10
    
    density_estimator =likelihood_nn(model=model, 
                                 hidden_features = hidden_features,
                                 num_transforms = num_transforms,
                                 num_bins = num_bins
                                 )

    
    prior_sbi = generate_prior()
    inference = SNLE(prior=prior_sbi, density_estimator=density_estimator)
    # inference = SNLE(prior=prior_sbi,)
    inference.append_simulations(theta_train, x_train)

    
    likelihood_estimator = inference.train(learning_rate = learning_rate,
                                           training_batch_size = training_batch_size,
                                           stop_after_epochs = patience,
                                           dataloader_kwargs= {"num_workers": 0}
    )

    
    
    val = inference._summary["best_validation_log_prob"][0]
    
    
    del inference
    del likelihood_estimator
    # del density_estimator
    gc.collect()
    
    return val


agama.setRandomSeed(13)
torch.manual_seed(13)
np.random.seed(13)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

import pickle
with open("./model_11/tune_model_14.pkl", "wb") as handle:
    pickle.dump(study, handle)
    
print(study.best_params)
