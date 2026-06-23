import os
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
import time
from standardization import standardize

from joblib import Parallel, delayed

# set agama unit to be in Msun, kpc, km/s
agama.setUnits(mass=1 * u.Msun, length=1*u.kpc, velocity=1 * u.km /u.s)

agama.setRandomSeed(13)
torch.manual_seed(13)
np.random.seed(13)
torch.set_num_threads(1)

def sample_single_galaxy(i, x_o, posterior):
    
    import warnings
    warnings.filterwarnings("ignore", message="An x with a batch size of")
    warnings.filterwarnings("ignore", message="As of sbi v0.19.0")
    
    
    start_time = time.perf_counter()
    print(f"starting {i}th galaxy")

    x_o = torch.from_numpy(x_o).float()
    
    with torch.no_grad():
        samples = posterior.sample((2000,), x=x_o, show_progress_bars=True).numpy()
    
    end_time = time.perf_counter()
    print(f"Galaxy {i} took {end_time - start_time:.4f} seconds")
    
    return i, samples

if __name__ == "__main__":
    
    # testing dataset
    # test_theta_raw = np.array(pd.read_csv("./model 3/test_theta_g1000_s1000.csv", header=None))
    prof = "core"
    
    # test_x_raw = np.array(pd.read_csv(f"./model 9/x_test_{prof}.csv", header=None))

    # indices = np.array(pd.read_csv("./model 9/theta_test.csv", header=None))
    # test_x = np.split(test_x_raw, np.cumsum(indices)[:-1], axis=0)
    
    test_theta_raw = np.array(pd.read_csv("./model_14/test_theta_model_14.csv", header=None))
    test_x_raw = np.array(pd.read_csv("./model_14/test_x_model_14.csv", header=None))

    t, x = standardize()
    
    test_x_raw = (test_x_raw - x[1])/ x[2]
    
    test_x = [np.array(pd.read_csv(f"./model_13_1/mass_density_x_{prof}_model_13_1.csv", header=None))]

    # with open(f'./model 9/inference_model_9_{prof}.pkl', 'rb') as file:
    #     # Load the object from the file
    #     inference = pickle.load(file)
    
    with open(f'./model_13_1/inference_model_13_1.pkl', 'rb') as file:
        # Load the object from the file
        inference = pickle.load(file)
        

    posterior = inference.build_posterior( 
                                        mcmc_method="slice_np_vectorized", 
                                        mcmc_parameters={"warmup_steps":500,
                                                            "num_chains":8,
                                                            "num_workers": 1,
                                                            "init_strategy": "sir",
                                                            "thin": 4})

    # Test first 100 galaxies

    test_x = test_x

    n_galaxies_at_once = 1
    
    print(f"Starting parallel MCMC for {len(test_x)} galaxies...")
    start_time = time.perf_counter()
    
    results = Parallel(n_jobs=n_galaxies_at_once, verbose=10)(
        delayed(sample_single_galaxy)(i, x_o, posterior) 
        for i, x_o in enumerate(test_x)
    )
    
    # Sort result
    results.sort(key=lambda x: x[0])
    final_samples = [res[1] for res in results]

    # with open(f"./model 9/samples_model_9_{prof}.pkl", "wb") as handle:
    #     pickle.dump(final_samples, handle)
    
    pd.DataFrame(final_samples[0]).to_csv(f"./model_13_1/mass_density_samples_model_13_1_{prof}.csv", header=None, index=None)
    
    end_time = time.perf_counter()
    print(f"Total time for {len(test_x)} galaxies: {end_time - start_time:.2f} seconds")
        