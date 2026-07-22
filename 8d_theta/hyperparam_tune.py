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
import optuna
import gc
from object_handler import save_pickle, load_pickle
from model import prep_data, prep_inference, train_model



def objective(trial):
        
    torch.set_num_threads(4)

    global train_theta
    global train_x
    
    # Learning
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    training_batch_size = 512
    
    
    # Normalizing flow
    model = trial.suggest_categorical("model", ["maf", "nsf"])
    hidden_features = trial.suggest_int("hidden_features", 32, 128)
    num_transforms = trial.suggest_int("num_transforms", 3, 12)
    num_bins = trial.suggest_int("num_bins", 4, 12)
    patience = 10
    
    density_estimator ={"model": model, 
                        "hidden_features": hidden_features,
                        "num_transforms": num_transforms,
                        "num_bins": num_bins
                        }

    inference = prep_inference(train_theta,
                               train_x,
                               likelihood_estimator_settings = density_estimator)
    
    
    train_settings = {"learning_rate": learning_rate,
                      "training_batch_size": training_batch_size,
                      "stop_after_epochs": patience,
                      "max_num_epochs": 100}
    
    inference = train_model(inference,
                            train_settings)

    val = inference._summary["best_validation_log_prob"][0]
    
    
    del inference
    gc.collect()
    
    return val


if __name__ == "__main__":
    agama.setRandomSeed(13)
    torch.manual_seed(13)
    np.random.seed(13)

    dim = 5

    print(dim)    
    train_theta, train_x = prep_data(f"./8d_theta/model_8/{dim}d/train_theta.h5",
                                     f"./8d_theta/model_8/{dim}d/train_x.h5",
                                     standardization=True,
                                     uncertainty=True)
    
    train_theta, train_x = train_theta[:200000], train_x[:200000]
    
    ### For P(v| x, y, sigma, theta) ###
    train_theta = torch.column_stack((train_theta, train_x[:, :2]))
    train_x = train_x[:, 2:]
    #######################################

    # study = load_pickle("./8d_theta/model_1/tune.pkl")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    save_pickle(study, f"./8d_theta/model_9/{dim}d/tune.pkl")


