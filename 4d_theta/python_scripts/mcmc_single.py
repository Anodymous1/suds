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
import pandas as pd
import pickle
from prior_generation import generate_prior
from galaxy_generation import generate_galaxy_multiple

# set agama unit to be in Msun, kpc, km/s
agama.setUnits(mass=1 * u.Msun, length=1*u.kpc, velocity=1 * u.km /u.s)

agama.setRandomSeed(13)
torch.manual_seed(13)
np.random.seed(13)

torch.set_num_threads(1)



log_p_0 = 7
log_r_s = 0
gamma = 1
r_star = 0.2
    
x_o = torch.tensor(np.array(pd.read_csv(f"./mcmc/stars_for_mass_density({gamma}).csv", header=None)))

with open('./model/inference(poisson).pkl', 'rb') as file:
    # Load the object from the file
    inference = pickle.load(file)
    

posterior = inference.build_posterior( 
                                      mcmc_method="slice_np_vectorized", 
                                      mcmc_parameters={"warmup_steps":200,
                                                        "num_chains":32,
                                                        "num_workers": 8,
                                                        "init_strategy": "sir",
                                                        "thin": 4})



samples = posterior.sample((1000,), x=x_o)

pd.DataFrame(samples).to_csv(f"./mcmc/mass_density_profile_1k_32({gamma})", index=False, header=False)