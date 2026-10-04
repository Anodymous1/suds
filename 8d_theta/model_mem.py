from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import ctypes
import gc
import resource
import time
from datetime import datetime

import agama
import numpy as np
import psutil
import torch
from astropy import units as u
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR

from sbi.inference import SNLE
from sbi.utils import likelihood_nn

from standardization import standardize
from object_handler import save_pickle, load_csv, load_galaxies, load_h5

torch.set_num_threads(4)

agama.setUnits(mass=1 * u.Msun, length=1 * u.kpc, velocity=1 * u.km / u.s)
agama.setRandomSeed(13)
torch.manual_seed(13)
np.random.seed(13)


def mem(tag: str = "") -> None:
    """Print current and peak resident memory (Linux)."""
    cur = psutil.Process(os.getpid()).memory_info().rss / 1e9
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[mem] {tag}: current {cur:.2f} GB | peak {peak:.2f} GB", flush=True)


def release_memory() -> None:
    """Free Python garbage and hand freed heap memory back to the OS."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass  # not glibc (e.g. macOS)


def prep_data(train_theta: str,
              train_x: str,
              standardization: bool = True,
              uncertainty: bool = False,
              dim: int = 5,
              cut: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load theta and x as single, contiguous float32 tensors.

    Same logic as before, but every intermediate copy is dropped as soon as
    it is no longer needed, and slices are made contiguous so they don't keep
    the full-size parent buffer alive.
    """
    if not uncertainty:
        theta, k = load_galaxies(train_theta, "Tensor")
        prepped_theta = torch.repeat_interleave(theta.float(), k, dim=0)
        del theta, k
    else:
        prepped_theta = (load_h5(train_theta, "theta", "Tensor") if ".h5" in train_theta
                         else load_csv(train_theta, "Tensor"))
        prepped_theta = prepped_theta.float()  # no-op if already float32

    x = load_h5(train_x, "x", "Tensor") if ".h5" in train_x else load_csv(train_x, "Tensor")
    x = x.float()
    release_memory()
    mem("after loading")

    # Reduce to the desired dimension (.contiguous() so the full buffer is freed)
    if dim != x.shape[1]:
        if dim == 4:
            x = x[:, :4].contiguous()
            if uncertainty:
                prepped_theta = prepped_theta[:, :10].contiguous()
        elif dim == 3:
            x = x[:, (0, 1, 4)]
            if uncertainty:
                prepped_theta = prepped_theta[:, (0, 1, 2, 3, 4, 5, 6, 7, 10)]

    if cut is not None:
        prepped_theta = prepped_theta[:cut].clone()
        x = x[:cut].clone()
    release_memory()

    if standardization:
        x = standardize(x)[0].float()
        release_memory()

    # sbi's append_simulations would reject NaN/inf; check it here instead
    if not torch.isfinite(x).all() or not torch.isfinite(prepped_theta).all():
        raise ValueError("theta or x contains NaN/inf values")

    mem("after prep_data")
    return prepped_theta, x


def build_net(theta: torch.Tensor,
              x: torch.Tensor,
              settings: dict | None,
              n_zscore: int = 1_000_000) -> torch.nn.Module:
    """
    Build the density estimator with sbi's own builder. Z-scoring statistics
    come from a random subsample instead of a copy of the full training set.
    """
    builder = likelihood_nn(**settings) if settings is not None else likelihood_nn(model="maf")
    sub = torch.randperm(len(x))[:n_zscore]
    return builder(theta[sub], x[sub])


def train_model(net: torch.nn.Module,
                theta: torch.Tensor,
                x: torch.Tensor,
                training_batch_size: int = 50,
                learning_rate: float = 5e-4,
                validation_fraction: float = 0.1,
                stop_after_epochs: int = 20,
                max_num_epochs: int = 100,
                clip_max_norm: float | None = 5.0,
                scheduler: bool = False,
                min_delta: float = 0.01,
                val_batch_size: int = 65536) -> tuple[torch.nn.Module, float, dict]:
    """
    Same training as sbi's SNLE.train() (loss = -log q(x | theta), Adam,
    grad clipping, drop_last, early stopping on validation log-prob), plus an
    optional cosine LR schedule. Batches are gathered by index tensors, so
    the dataset is never copied and no Python index lists are built.
    """
    start_time = time.perf_counter()
    print(f"{datetime.now()}: Beginning model training", flush=True)

    n = len(x)
    n_val = int(validation_fraction * n)
    perm = torch.randperm(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    sched = CosineAnnealingLR(opt, T_max=max_num_epochs) if scheduler else None

    best_val, best_epoch = -float("inf"), 0
    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    summary = {"training_log_probs": [], "validation_log_probs": []}

    for epoch in range(max_num_epochs):
        # ---- train ----
        net.train()
        train_sum, train_count = 0.0, 0
        shuffled = train_idx[torch.randperm(len(train_idx))]
        for idx in shuffled.split(training_batch_size):
            if len(idx) < training_batch_size:
                continue  # drop_last, as in sbi
            log_prob = net.log_prob(x[idx], context=theta[idx])
            loss = -log_prob.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip_max_norm is not None:
                clip_grad_norm_(net.parameters(), max_norm=clip_max_norm)
            opt.step()
            train_sum += log_prob.detach().sum().item()
            train_count += len(idx)
        del shuffled
        if sched is not None:
            sched.step()

        # ---- validate ----
        net.eval()
        with torch.no_grad():
            val_sum = sum(net.log_prob(x[i], context=theta[i]).sum().item()
                          for i in val_idx.split(val_batch_size))
        val = val_sum / n_val
        summary["training_log_probs"].append(train_sum / max(train_count, 1))
        summary["validation_log_probs"].append(val)

        if val > best_val + min_delta:
            best_val, best_epoch = val, epoch
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

        print(f"epoch {epoch}: train {summary['training_log_probs'][-1]:.4f} | "
              f"val {val:.4f} | best {best_val:.4f} (epoch {best_epoch}) | "
              f"lr {opt.param_groups[0]['lr']:.2e}", flush=True)
        if epoch == 0:
            mem("after first epoch")

        if epoch - best_epoch >= stop_after_epochs:
            print(f"stopping at epoch {epoch}; best {best_val:.4f} at epoch {best_epoch}")
            break

    net.load_state_dict(best_state)
    net.zero_grad(set_to_none=True)
    print(f"Training took {time.perf_counter() - start_time:.1f} s")
    return net, best_val, summary


def to_inference(net: torch.nn.Module,
                 x_dim: int,
                 settings: dict | None,
                 best_val: float,
                 summary: dict) -> SNLE:
    """
    Wrap the trained net in an SNLE object holding no training data, so
    downstream code can still call inference.build_posterior(prior=...).
    """
    builder = likelihood_nn(**settings) if settings is not None else likelihood_nn(model="maf")
    inference = SNLE(density_estimator=builder)
    inference._neural_net = net
    inference._x_shape = torch.Size([1, x_dim])  # normally set inside .train()
    inference._best_val_log_prob = best_val
    inference._summary["validation_log_probs"] = summary["validation_log_probs"]
    inference._summary["training_log_probs"] = summary["training_log_probs"]
    return inference


if __name__ == "__main__":
    dim = '3'
    model = "Ting"
    mem("start")
    train_theta, train_x = prep_data(f"./8d_theta/model_{model}/small_unc/train_theta.h5",
                                     f"./8d_theta/model_{model}/small_unc/train_x.h5",
                                     uncertainty=True,
                                     dim=dim,)
                                     # cut=100000)

    likelihood_estimator_settings = {'model': 'maf',
                                    'hidden_features': 71,
                                    'num_transforms': 8,
                                    'num_bins': 4}

    net = build_net(train_theta, train_x, likelihood_estimator_settings)
    mem("after building net")

    net, best_val, summary = train_model(
        net, train_theta, train_x,
        training_batch_size=2048,
        learning_rate=0.0005,
        validation_fraction=0.1,
        stop_after_epochs=20,
        max_num_epochs=200,
        clip_max_norm=3.0,
        scheduler=True,
    )

    x_dim = train_x.shape[1]
    del train_theta, train_x
    release_memory()

    inference = to_inference(net, x_dim, likelihood_estimator_settings, best_val, summary)
    save_pickle(inference, f"./8d_theta/model_{model}/small_unc/inference.pkl", override=False)
    mem("end")