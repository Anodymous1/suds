import os
os.environ["OMP_NUM_THREADS"] = "1"     # agama
os.environ["MKL_NUM_THREADS"] = "1"     # numpy, scipy
os.environ["OPENBLAS_NUM_THREADS"] = "1"    # numpy
os.environ["NUMEXPR_NUM_THREADS"] = "1"     # pandas

import torch 
import numpy as np

import pandas as pd
from galaxy_generation import generate_galaxy_multiple
from prior_generation import generate_prior


torch.set_num_threads(1)

num_galaxies = 100_000
prior = generate_prior()
theta = prior.sample((num_galaxies,))
n_stars = np.random.poisson(100, size=num_galaxies)

theta = torch.repeat_interleave(theta, torch.tensor(n_stars), dim=0)
x = generate_galaxy_multiple(theta, n_stars, 25)

pd.DataFrame(x).to_csv("train_x_g100k.csv", index=None, header=None)
pd.DataFrame(theta).to_csv("train_theta_g100k.csv", index=None, header=None)