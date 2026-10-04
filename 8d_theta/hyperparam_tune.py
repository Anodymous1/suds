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
# from model import prep_data, prep_inference, train_model
from model_mem import prep_data, build_net, train_model, release_memory
from memory_watchdog import start_memory_watchdog



def objective(trial):
        
    torch.set_num_threads(4)

    global train_theta
    global train_x
    
    # Learning
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    training_batch_size = trial.suggest_int("batch_size", 256, 8192, log=True)
    
    
    # Normalizing flow
    model = trial.suggest_categorical("model", ["maf", "nsf"])
    hidden_features = trial.suggest_int("hidden_features", 32, 128)
    num_transforms = trial.suggest_int("num_transforms", 3, 12)
    num_bins = trial.suggest_int("num_bins", 4, 12)
    patience = 20
    
    likelihood_estimator_settings ={"model": model, 
                                    "hidden_features": hidden_features,
                                    "num_transforms": num_transforms,
                                    "num_bins": num_bins,
                                    }

    net = build_net(train_theta, train_x, likelihood_estimator_settings)
    

    
    net, best_val, summary = train_model(
        net, train_theta, train_x,
        training_batch_size=training_batch_size,
        learning_rate=learning_rate,
        validation_fraction=0.1,
        stop_after_epochs=patience,
        max_num_epochs=200,
        clip_max_norm=3.0,
        scheduler=True,
    )

    del net
    gc.collect()
    release_memory()
    
    return best_val


if __name__ == "__main__":
    agama.setRandomSeed(13)
    torch.manual_seed(13)
    np.random.seed(13)

    start_memory_watchdog(limit_gb=15, min_available_gb=4, interval=1.0, log_every=5)
    dim = 3
    model = "14"

    print(dim)    
    train_theta, train_x = prep_data(f"./8d_theta/model_{model}/{dim}d/train_theta.h5",
                                     f"./8d_theta/model_{model}/{dim}d/train_x.h5",
                                     standardization=True,
                                     uncertainty=True,
                                     dim=dim)
    
    train_theta, train_x = train_theta[:500000], train_x[:500000]
    
    # ### For P(v| x, y, sigma, theta) ###
    # train_theta = torch.column_stack((train_theta, train_x[:, :2]))
    # train_x = train_x[:, 2:]
    #######################################
    print(train_theta.shape, train_x.shape)

    # study = load_pickle("./8d_theta/model_1/tune.pkl")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    save_pickle(study, f"./8d_theta/model_14/{dim}d/tune.pkl")


