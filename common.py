import os
import json
import math
import pickle
import argparse
from datetime import datetime
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader

import wandb
import yaml
import sys
from dotwiz import DotWiz


class RatWindowDataset(Dataset):
    def __init__(self, data_list, hist_len, future_len, data_time_list=None, return_idx=False, file_names=None):
        self.start_ind = []
        self.time_int = []
        for i, d in enumerate(data_list):
            T = d.shape[0]
            L = hist_len + future_len
            n_windows = T - L + 1
            for s in range(n_windows):
                self.start_ind.append((i, s))
                if data_time_list is not None:
                    # (t_hist_start, t_hist_end, t_fut_start, t_fut_end)
                    self.time_int.append((data_time_list[i][s], data_time_list[i][s + hist_len - 1], data_time_list[i][s + hist_len], data_time_list[i][s + L - 1]))

        self.data_list = data_list
        self.hist_len = hist_len
        self.future_len = future_len
        self.return_idx = return_idx
        self.file_names = file_names

    def __len__(self):
        return len(self.start_ind)

    def __getitem__(self, idx):
        fi, s = self.start_ind[idx]
        d = self.data_list[fi]

        hist = d[s: s + self.hist_len].astype(np.float32)
        fut = d[s + self.hist_len: s + self.hist_len + self.future_len].astype(np.float32)

        sample = {
            "hist": torch.from_numpy(hist),
            "future": torch.from_numpy(fut),
        }
        if self.file_names is not None:
            sample["file_name"] = self.file_names[fi]
        if self.return_idx:
            sample["file_index"] = fi
            sample["start_index"] = s
            if self.time_int:
                sample["time_interval"] = self.time_int[idx]
        return sample


def load_dset(args):
    """
    Build train/test datasets from pickled segments.
    Expect each pkl to provide a dict with:
      - 'segment': [T, 2, P, N_mice] (example) or similar
      - 'time':    [T]
    We select a target point index (idx_target) and downsample in time.
    """
    if isinstance(args, dict):
        args = DotWiz(args)

    idx_target = args.idx_target
    hist_len = args.hist_len
    future_len = args.future_len

    train_pkl_list = args.train_pkl
    test_pkl_list = args.test_pkl

    train_data_raw = []
    for pkl_path in train_pkl_list:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        train_data_raw.append(data)

    test_data_raw = []
    for pkl_path in test_pkl_list:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        test_data_raw.append(data)

    print(f"Hist len: {hist_len}, Fut len: {future_len}")

    train_data_downsampled = []
    train_time_downsampled = []
    for i, d in enumerate(train_data_raw):
        segment = d["segment"]
        time = d["time"]
        segment_ds = segment[::args.downsample_factor][:, :, :, idx_target]
        time_ds = time[::args.downsample_factor]
        train_data_downsampled.append(segment_ds)
        train_time_downsampled.append(time_ds)
        print(f"Train segment {i} ds shape: {segment_ds.shape}")

    train_time_diffs = [np.diff(t) for t in train_time_downsampled]
    train_time_diffs = np.concatenate(train_time_diffs, axis=0)
    train_time_avg_dt = np.mean(train_time_diffs)
    print(f"Train avg dt after downsampling: {train_time_avg_dt:.6f} s")
    print(f"Train Hertz after downsampling: {1.0 / train_time_avg_dt:.2f} Hz")
    print(f"Train hist window length (s): {hist_len * train_time_avg_dt:.2f} s")
    print(f"Train future window length (s): {future_len * train_time_avg_dt:.2f} s")

    test_data_downsampled = []
    test_time_downsampled = []
    for i, d in enumerate(test_data_raw):
        segment = d["segment"]
        time = d["time"]
        segment_ds = segment[::args.downsample_factor][:, :, :, idx_target]
        time_ds = time[::args.downsample_factor]
        test_data_downsampled.append(segment_ds)
        test_time_downsampled.append(time_ds)
        print(f"Test segment {i} ds shape: {segment_ds.shape}")

    test_time_diffs = [np.diff(t) for t in test_time_downsampled]
    test_time_diffs = np.concatenate(test_time_diffs, axis=0)
    test_time_avg_dt = np.mean(test_time_diffs)
    print(f"Test avg dt after downsampling: {test_time_avg_dt:.6f} s")
    print(f"Test Hertz after downsampling: {1.0 / test_time_avg_dt:.2f} Hz")
    print(f"Test hist window length (s): {hist_len * test_time_avg_dt:.2f} s")
    print(f"Test future window length (s): {future_len * test_time_avg_dt:.2f} s")

    assert train_data_downsampled[0].shape[1] == 1, "Expected number of mice = 1 in dim=1."
    assert test_data_downsampled[0].shape[1] == 1, "Expected number of mice = 1 in dim=1."
    train_data_downsampled = [d.squeeze(1) for d in train_data_downsampled]
    test_data_downsampled = [d.squeeze(1) for d in test_data_downsampled]

    train_ds = RatWindowDataset(train_data_downsampled, hist_len, future_len)
    test_ds = RatWindowDataset(test_data_downsampled, hist_len, future_len)
    return train_ds, test_ds


# -----------------------------------------------------------------------------
# Masks & NaN handling
# -----------------------------------------------------------------------------
def _mask_from_xy(x: torch.Tensor) -> torch.Tensor:
    """
    Build validity mask from an array that may be [B,T,2] or [B,T,2,P].
    Returns [B,T] (True=valid).
    """
    if x.dim() == 3:          # [B, T, 2]
        return torch.isfinite(x).all(dim=-1)
    elif x.dim() == 4:        # [B, T, 2, P]
        return torch.isfinite(x).all(dim=(-1, -2))
    else:
        raise ValueError(f"Unexpected shape for mask: {x.shape}")


def make_hist_mask(x: torch.Tensor) -> torch.Tensor:
    return _mask_from_xy(x)


def make_future_mask(x: torch.Tensor) -> torch.Tensor:
    return _mask_from_xy(x)


def fill_nan(x: torch.Tensor, value: float = 0.0) -> torch.Tensor:
    """
    Replace NaN/Inf entries with a given constant (default 0.0).
    """
    m = torch.isfinite(x)
    return torch.where(m, x, torch.full_like(x, value))


def masked_gaussian_nll_per_mode(mean, logvar, target, valid):
    """
    Compute masked Gaussian negative log-likelihood per mode.

    Args:
        mean:   [B, K, T, 2]   predicted means
        logvar: [B, K, T, 2]   predicted log-variances log(σ^2) per coord
        target: [B, T, 2]      ground-truth positions
        valid:  [B, T]         True for valid timesteps

    Returns:
        nll_sum: [B, K]  -- NLL summed over (T, 2) with mask
        nll_avg: [B, K]  -- NLL averaged over valid timesteps (for reporting)
    """
    var = logvar.exp()
    diff = target.unsqueeze(1) - mean
    
    nll = 0.5 * (diff * diff / var + logvar)
    nll = nll.sum(dim=-1)
    
    tmask = valid.unsqueeze(1).float()
    nll_sum = (nll * tmask).sum(dim=-1)
    denom = tmask.sum(dim=-1).clamp_min(1.0)
    nll_avg = nll_sum / denom

    return nll_sum, nll_avg


def closest_mode_by_l2(mean, target, valid):
    """
    Select the "closest" mode in the Wayformer / MultiPath++ sense:
    the mode whose *mean trajectory* is closest to the ground truth trajectory,
    measured by average L2 distance over valid timesteps.

    Args:
        mean:   [B, K, T, 2]   predicted means
        target: [B, T, 2]      ground-truth positions
        valid:  [B, T]         True for valid timesteps

    Returns:
        closest:      [B]      index of closest mode per sample
        l2_avg_modes: [B, K]   average L2 distance per mode (for logging)
    """
    diff = mean - target.unsqueeze(1)

    l2 = torch.sqrt((diff * diff).sum(dim=-1))

    tmask = valid.unsqueeze(1).float()
    l2_sum = (l2 * tmask).sum(dim=-1)
    denom = tmask.sum(dim=-1).clamp_min(1.0)
    l2_avg = l2_sum / denom

    closest = torch.argmin(l2_avg, dim=1)

    return closest, l2_avg


def wayformer_loss(conf_logits, mean, logvar, target, valid):
    """
    Wayformer-style objective (following MultiPath / MultiPath++):
        1) Choose the "correct" mode as the one whose *mean trajectory* is
           closest to the ground truth (average L2 distance over time).
        2) Classification loss:   log Pr(î | Y)    -> CrossEntropy
        3) Regression loss:       log Pr(G | T_î)  -> Gaussian NLL on chosen mode

        Total loss = CE + NLL_chosen_mode  (we minimize this).

    Args:
        conf_logits: [B, K]        unnormalized logits per mode
        mean/logvar: [B, K, T, 2]  predicted Gaussian params (mean, log σ^2)
        target:      [B, T, 2]     ground-truth positions
        valid:       [B, T]        True for valid timesteps

    Returns:
        dict with:
          'loss'      : scalar total loss
          'ce'        : classification loss (CE)
          'nll_best'  : regression NLL on chosen mode (mean over batch)
          'avg_nll'   : average NLL over all modes (for monitoring)
          'l2_best'   : average L2 distance of chosen mode (for monitoring)
    """
    B, K, T, _ = mean.shape

    closest, l2_avg = closest_mode_by_l2(mean, target, valid)
    ce = F.cross_entropy(conf_logits, closest)

    nll_sum, nll_avg = masked_gaussian_nll_per_mode(
        mean, logvar, target, valid
    )
    
    row_idx = torch.arange(B, device=closest.device)
    nll_best = nll_avg[row_idx, closest].mean()
    nll_per_data = nll_sum[row_idx, closest]
    
    avg_nll = nll_avg.mean()
    l2_best = l2_avg[row_idx, closest].mean()

    loss = ce + nll_best

    return {
        "loss": loss,
        "ce": ce,
        "nll_best": nll_best,
        "avg_nll": avg_nll,
        "l2_best": l2_best,
        "nll_per_data": nll_per_data,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_nll_best = 0.0
    total_avg_nll = 0.0
    total_l2_best = 0.0
    n_samples = 0

    total_ADE = 0.0
    total_FDE = 0.0

    for batch in loader:
        hist = batch["hist"].to(device=device, dtype=torch.float32)
        fut = batch["future"].to(device=device, dtype=torch.float32)

        hist_mask = make_hist_mask(hist)
        fut_mask = make_future_mask(fut)
        hist = fill_nan(hist, 0.0)
        fut = fill_nan(fut, 0.0)

        if hist.dim() == 4:
            hist = hist.mean(dim=-1)
        if fut.dim() == 4:
            fut = fut.mean(dim=-1)

        conf_logits, mean, logvar = model(hist, hist_mask)
        loss_dict = wayformer_loss(conf_logits, mean, logvar, fut, valid=fut_mask)
        batch_sz = hist.size(0)
        n_samples += batch_sz

        total_loss += loss_dict["loss"].item() * batch_sz
        total_ce += loss_dict["ce"].item() * batch_sz
        total_nll_best += loss_dict["nll_best"].item() * batch_sz
        total_avg_nll += loss_dict["avg_nll"].item() * batch_sz
        total_l2_best += loss_dict["l2_best"].item() * batch_sz

        best_idx, _ = closest_mode_by_l2(mean, fut, fut_mask)
        bidx = best_idx.view(-1, 1, 1, 1).expand(-1, 1, fut.size(1), 2)
        best_mean = mean.gather(dim=1, index=bidx).squeeze(1)

        l2 = torch.sqrt(((best_mean - fut) ** 2).sum(dim=-1))
        ade = (l2 * fut_mask).sum(dim=1) / fut_mask.sum(dim=1).clamp(min=1)
        fde = l2[:, -1]
        total_ADE += ade.sum().item()
        total_FDE += fde.sum().item()

    avg_loss = total_loss / max(n_samples, 1)
    avg_ce = total_ce / max(n_samples, 1)
    avg_nll_best = total_nll_best / max(n_samples, 1)
    avg_avg_nll = total_avg_nll / max(n_samples, 1)
    avg_l2_best = total_l2_best / max(n_samples, 1)
    avg_ADE = total_ADE / max(n_samples, 1)
    avg_FDE = total_FDE / max(n_samples, 1)

    return {
        "val_loss": avg_loss,
        "val_ce": avg_ce,
        "val_nll_best": avg_nll_best,
        "val_avg_nll": avg_avg_nll,
        "val_l2_best": avg_l2_best,
        "val_ADE": avg_ADE,
        "val_FDE": avg_FDE,
    }
