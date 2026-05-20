import agama
import torch 
import numpy as np
from astropy import units as u
import pandas as pd
import pickle



# set agama unit to be in Msun, kpc, km/s
agama.setUnits(mass=1 * u.Msun, length=1*u.kpc, velocity=1 * u.km /u.s)

agama.setRandomSeed(13)
torch.manual_seed(13)
np.random.seed(13)
torch.set_num_threads(1)
import os
os.environ["OMP_NUM_THREADS"] = "1"



test_theta_raw = np.array(pd.read_csv("./data/test_theta.csv", header=None))
test_x_raw = np.array(pd.read_csv("./data/test_x.csv", header=None))

test_theta, index = np.unique(test_theta_raw, axis=0, return_index=True)

index = np.sort(index)

test_x = np.split(test_x_raw, index, axis=0)[1:]


with open('./model/likelihood_estimator.pkl', 'rb') as file:
    # Load the object from the file
    likelihood_estimator = pickle.load(file)
    

with open('./model/inference.pkl', 'rb') as file:
    # Load the object from the file
    inference = pickle.load(file)
    

posterior = inference.build_posterior(likelihood_estimator, 
                                      mcmc_method="slice_np_vectorized", 
                                      mcmc_parameters={"warmup_steps":200,
                                                        "num_chains":8,
                                                        "num_workers": 4,
                                                        "init_strategy": "sir"})

# Test first 100 galaxies
test_x = test_x[:100]

samples = [posterior.sample((100,), x=x_o) for x_o in test_x]

with open("./mcmc/samples(100).pkl", "wb") as handle:
    pickle.dump(samples, handle)