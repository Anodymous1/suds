import os
os.environ["OMP_NUM_THREADS"] = "1"     # agama
os.environ["MKL_NUM_THREADS"] = "1"     # numpy, scipy
os.environ["OPENBLAS_NUM_THREADS"] = "1"    # numpy
os.environ["NUMEXPR_NUM_THREADS"] = "1"     # pandas
import random
import agama
import torch 
import numpy as np

import pandas as pd
import pickle
from galaxy_generation import generate_galaxy_multiple
import time
from sbi.inference import SNLE
from prior_generation import generate_prior
from standardization import get_standard
# from sbi.inference.posteriors.mcmc_posterior import MCMCPosterior

import sbi
print(sbi.__version__)

agama.setRandomSeed(13)
torch.manual_seed(13)
np.random.seed(13)
random.seed(13)

import corner

torch.set_num_threads(4)

# Galaxy
log_p_0 = 7
log_r_s = 0
gamma = 0
r_star_div_r_s = 0.2
r_star = r_star_div_r_s * 10 ** log_r_s

if gamma == 0:
    prof = "core"
else:
    prof = "cusp"


galaxy = np.expand_dims(np.array([log_p_0, log_r_s, gamma, r_star_div_r_s]), axis=0).astype(np.float32)



# MCMC settings
mcmc_method="slice_np_vectorized" 
mcmc_parameters={"warmup_steps":500,
                    "num_chains":32,
                    "num_workers": 1,
                    "init_strategy": "sir",
                    "thin": 4}

# x_o
x_o = torch.tensor(np.array(pd.read_csv(f"./model_12/x_o_{prof}.csv", header=None))).float()
t, x = get_standard()
x_o = ((x_o - x[1]) / x[2]).float()

# Training
# Train model 3
# Sequential training settings
num_samples = 1000
num_rounds = 5
base_lr = 0.00015197901214187632
base_batch_size = 256
base_stop_after_epochs = 10

theta_train = torch.from_numpy(np.array(pd.read_csv("./model_11/train_theta.csv", header=None))).float()
x_train = torch.from_numpy(np.array(pd.read_csv("./model_11/train_x.csv", header=None))).float()
x_train = ((x_train - x[1]) / x[2]).float()

inference = SNLE(prior=generate_prior())
inference.append_simulations(theta_train, x_train)

initial_dataset_size = len(inference.get_simulations()[0]) 

inference.train(training_batch_size=base_batch_size, 
                learning_rate=base_lr, 
                stop_after_epochs=base_stop_after_epochs)

# Initialize loop
posterior = inference.build_posterior(mcmc_method=mcmc_method, mcmc_parameters=mcmc_parameters)
proposal = posterior.set_default_x(x_o)

# Begin training
start_time_whole = time.perf_counter()

for i in range(num_rounds):
    print(f"Beginning round {i}")
    
    # # Train settings
    # current_lr = base_lr * (0.8 ** i)
    
    # current_dataset_size = len(inference.get_simulations()[0])
    # current_batch_size = int(base_batch_size * (current_dataset_size / initial_dataset_size))

    # current_stop_after_epochs = base_stop_after_epochs + i * 2
    
    # arg = {
    #         "training_batch_size": current_batch_size,
    #         "learning_rate": current_lr,
    #         "validation_fraction": 0.1,
    #         "stop_after_epochs": current_stop_after_epochs,
    #         "max_num_epochs": 2**31 - 1,
    #         "clip_max_norm": 5.0,
    #         "resume_training": False,
    #         "discard_prior_samples": False,
    #         "retrain_from_scratch": True,
    #         "show_train_summary": True,
    #         # "dataloader_kwargs": {"num_workers": 2, 
    #         #                         "persistent_workers": True}
    # }
    
    arg = {
        "training_batch_size": base_batch_size,
        "learning_rate": base_lr,
        "validation_fraction": 0.1,
        "stop_after_epochs": base_stop_after_epochs,
        "max_num_epochs": 2**31 - 1,
        "clip_max_norm": 5.0,
        "resume_training": False,
        "discard_prior_samples": False,
        "retrain_from_scratch": True,
        "show_train_summary": True,
        # "dataloader_kwargs": {"num_workers": 2, 
        #                         "persistent_workers": True}
    }
    
    start_time = time.perf_counter()
    
    theta = proposal.sample((num_samples,))
    
    n_stars = torch.tensor(np.random.poisson(100, size=num_samples))
    new_x = generate_galaxy_multiple(theta, n_stars, 4)
    new_x = ((new_x - x[1]) / x[2]).float()
    new_theta = torch.repeat_interleave(theta, n_stars, dim=0)
    
    inference.append_simulations(new_theta, new_x).train(**arg)
    
    posterior = inference.build_posterior(mcmc_method=mcmc_method, mcmc_parameters=mcmc_parameters)
    proposal = posterior.set_default_x(x_o)
    
    end_time = time.perf_counter()
    print(f"Round {i} took {end_time - start_time:.4f} seconds")

end_time_whole = time.perf_counter()
print(f"Entire training took {end_time_whole - start_time_whole:.4f} seconds")

# save model
with open(f"./model_12/inference_model_12_{prof}.pkl", "wb") as handle:
    pickle.dump(inference, handle)