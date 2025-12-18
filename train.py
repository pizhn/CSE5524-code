import os
import json
import math
import pickle
import argparse
import datetime
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

import argparse
import os
import json
import time
import logging
from collections import defaultdict, deque

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from common import RatWindowDataset
from model import RatTransformer

# 1. Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--idx_target", type=int, nargs="+", default=[3], help="Indices of keypoints to train on")
parser.add_argument("--train_pkl", type=str, nargs="+")
parser.add_argument("--test_pkl", type=str, nargs="+")
parser.add_argument("--d_model", type=int, default=64)
parser.add_argument("--n_head", type=int, default=4)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--downsample_factor", type=int, default=3)
parser.add_argument("--hist_len", type=int, default=110)
parser.add_argument("--future_len", type=int, default=33)
parser.add_argument("--n_modes", type=int, default=4)
parser.add_argument("--epochs", type=int, default=2000)
parser.add_argument("--train_batch_size", type=int, default=64)
parser.add_argument("--eval_batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=0.002)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.999)
parser.add_argument("--eta_min", type=float, default=1e-5)
parser.add_argument("--independent_modes", action="store_true", help="If set, use independent modes.")
parser.add_argument("--save_every_epochs", type=int, default=50)
parser.add_argument("--config", help="Path to YAML config file")
parser.add_argument("--debug", action="store_true", help="If set, run in debug mode.")

# New argument for logging frequency
parser.add_argument("--print_freq", type=int, default=10, help="Log every N iterations")

args = parser.parse_args()

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load YAML config and override defaults
if args.config:
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Detect CLI-provided option names to avoid overwriting them with config file values
    cli_provided = set()
    for tok in sys.argv[1:]:
        if tok.startswith("--"):
            key = tok.lstrip("-").split("=")[0]
            cli_provided.add(key)

    for k, v in config.items():
        if k in cli_provided:
            logger.info(f"CLI override present for '{k}', keeping CLI value.")
            continue
        setattr(args, k, v)

# -----------------------------------------------------------------------------
# Logging / checkpoint paths
# -----------------------------------------------------------------------------
time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# ckpt_save_path = os.path.join("./sequence_ckpt", f"mouse_seq_pred_{time_str}")
ckpt_save_path = os.path.join("./sequence_ckpt", args.save_path, f"{time_str}")
os.makedirs(ckpt_save_path, exist_ok=True)
log_file = os.path.join(ckpt_save_path, "log.txt")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh = logging.FileHandler(log_file)
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info("Logger initialized.")
logger.info(f"Checkpoint directory: {ckpt_save_path}")

# Save args snapshot
with open(os.path.join(ckpt_save_path, "args.json"), "w") as f:
    json.dump(vars(args), f, indent=4)

def load_dset(args):
    """
    Build train/test datasets from pickled segments.
    Expect each pkl to provide a dict with:
      - 'segment': [T, 2, P, N_mice] (example) or similar
      - 'time':    [T]
    We select a target point index (idx_target) and downsample in time.
    """
    idx_target = args.idx_target
    hist_len = args.hist_len
    future_len = args.future_len

    train_pkl_list = args.train_pkl
    test_pkl_list = args.test_pkl

    # Load raw train segments
    train_data_raw = []
    for pkl_path in train_pkl_list:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        train_data_raw.append(data)

    # Load raw test segments
    test_data_raw = []
    for pkl_path in test_pkl_list:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        test_data_raw.append(data)

    print(f"Hist len: {hist_len}, Fut len: {future_len}")

    # Downsample and select target point/channel for train
    train_data_downsampled = []
    train_time_downsampled = []
    for i, d in enumerate(train_data_raw):
        segment = d["segment"]  # expected shape: [T, 2, P, N_mice] or [T, 2, P]
        time = d["time"]        # [T]
        # Select target mouse index and downsample in time
        segment_ds = segment[::args.downsample_factor][:, :, :, idx_target]  # -> [T_ds, 2, P] (if N_mice dim present)
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

    # Downsample and select target point/channel for test
    test_data_downsampled = []
    test_time_downsampled = []
    for i, d in enumerate(test_data_raw):
        segment = d["segment"]
        time = d["time"]
        segment_ds = segment[::args.downsample_factor][:, :, :, idx_target]  # -> [T_ds, 2, P]
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

    # If data still has a "num_mice" dimension of size 1 (e.g. [T, 1, 2, P]),
    # you can squeeze it here. Adjust to your actual raw shapes.
    # Below assertions assume the second dim is number of mice and equals 1.
    assert train_data_downsampled[0].shape[1] == 1, "Expected number of mice = 1 in dim=1."
    assert test_data_downsampled[0].shape[1] == 1, "Expected number of mice = 1 in dim=1."
    train_data_downsampled = [d.squeeze(1) for d in train_data_downsampled]  # -> [T, 2, P]
    test_data_downsampled = [d.squeeze(1) for d in test_data_downsampled]    # -> [T, 2, P]


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


# -----------------------------------------------------------------------------
# Distance and NLL helpers in Wayformer / MultiPath++ style
# -----------------------------------------------------------------------------
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
    # var > 0, since model clamps logvar to a sane range
    var = logvar.exp()                               # [B, K, T, 2]
    diff = target.unsqueeze(1) - mean                # [B, K, T, 2]

    # Per-timestep, per-mode NLL (dropping constant 0.5*log(2π))
    # log N(x; μ, σ^2) = -0.5 * ((x-μ)^2 / σ^2 + log σ^2 + const)
    # Here we return positive NLL = 0.5 * ((x-μ)^2 / σ^2 + log σ^2).
    nll = 0.5 * (diff * diff / var + logvar)         # [B, K, T, 2]
    nll = nll.sum(dim=-1)                            # [B, K, T]

    # Mask over time dimension
    tmask = valid.unsqueeze(1).float()               # [B, 1, T]
    nll_sum = (nll * tmask).sum(dim=-1)              # [B, K]
    denom = tmask.sum(dim=-1).clamp_min(1.0)         # [B, 1]
    nll_avg = nll_sum / denom                        # [B, K]

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
    # [B, K, T, 2]
    diff = mean - target.unsqueeze(1)

    # Per-timestep L2 distance
    l2 = torch.sqrt((diff * diff).sum(dim=-1))       # [B, K, T]

    # Mask over time
    tmask = valid.unsqueeze(1).float()               # [B, 1, T]
    l2_sum = (l2 * tmask).sum(dim=-1)                # [B, K]
    denom = tmask.sum(dim=-1).clamp_min(1.0)         # [B, 1]
    l2_avg = l2_sum / denom                          # [B, K]

    # Closest mode = argmin over mean trajectory distance (MultiPath/Wayformer)
    closest = torch.argmin(l2_avg, dim=1)            # [B]

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

    # 1) Choose closest mode based on mean L2 trajectory distance
    closest, l2_avg = closest_mode_by_l2(mean, target, valid)   # [B], [B, K]

    # 2) Classification term: Cross-Entropy on conf_logits with target = closest mode
    ce = F.cross_entropy(conf_logits, closest)                  # scalar

    # 3) Regression term: Gaussian NLL on chosen mode
    nll_sum, nll_avg = masked_gaussian_nll_per_mode(
        mean, logvar, target, valid
    )                                                           # [B, K], [B, K]

    row_idx = torch.arange(B, device=closest.device)
    nll_best = nll_avg[row_idx, closest].mean()                 # scalar

    # For logging: average NLL over all modes, and L2 of chosen mode
    avg_nll = nll_avg.mean()
    l2_best = l2_avg[row_idx, closest].mean()

    loss = ce + nll_best

    return {
        "loss": loss,
        "ce": ce,
        "nll_best": nll_best,
        "avg_nll": avg_nll,
        "l2_best": l2_best,
    }


# -----------------------------------------------------------------------------
# Evaluation (loss + ADE/FDE on best-L2 mode)
# -----------------------------------------------------------------------------
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
        hist = batch["hist"].to(device=device, dtype=torch.float32)   # [B, Th, 2] or [B, Th, 2, P]
        fut = batch["future"].to(device=device, dtype=torch.float32)  # [B, H,  2] or [B, H,  2, P]

        hist_mask = make_hist_mask(hist)  # [B, Th]
        fut_mask = make_future_mask(fut)  # [B, H]

        hist = fill_nan(hist, 0.0)
        fut = fill_nan(fut, 0.0)

        # If input has an extra point dimension P, average over it before feeding the model
        if hist.dim() == 4:  # [B, Th, 2, P] -> [B, Th, 2]
            hist = hist.mean(dim=-1)
        if fut.dim() == 4:   # [B, H, 2, P] -> [B, H, 2]
            fut = fut.mean(dim=-1)

        conf_logits, mean, logvar = model(hist, hist_mask)  # mean/logvar: [B,K,H,2]
        loss_dict = wayformer_loss(conf_logits, mean, logvar, fut, valid=fut_mask)
        batch_sz = hist.size(0)
        n_samples += batch_sz

        total_loss += loss_dict["loss"].item() * batch_sz
        total_ce += loss_dict["ce"].item() * batch_sz
        total_nll_best += loss_dict["nll_best"].item() * batch_sz
        total_avg_nll += loss_dict["avg_nll"].item() * batch_sz
        total_l2_best += loss_dict["l2_best"].item() * batch_sz

        # ADE / FDE on best mode (by mean L2 trajectory distance,
        # consistent with Wayformer / MultiPath++ training and minADE/minFDE metrics)
        best_idx, _ = closest_mode_by_l2(mean, fut, fut_mask)         # [B]
        bidx = best_idx.view(-1, 1, 1, 1).expand(-1, 1, fut.size(1), 2)   # [B,1,H,2]
        best_mean = mean.gather(dim=1, index=bidx).squeeze(1)             # [B,H,2]

        l2 = torch.sqrt(((best_mean - fut) ** 2).sum(dim=-1))  # [B,H]
        ade = (l2 * fut_mask).sum(dim=1) / fut_mask.sum(dim=1).clamp(min=1)  # [B]
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


# -----------------------------------------------------------------------------
# Helper Class for Metric Smoothing
# -----------------------------------------------------------------------------
class SmoothedValue:
    """
    Track a series of values and provide access to smoothed values over a window 
    or the global average. Useful for logging loss during training.
    """
    def __init__(self, window_size=20, fmt="{median:.4f} ({global_avg:.4f})"):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """Returns the average of the values in the current window."""
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        """Returns the global average since the start of the epoch."""
        return self.total / self.count

    @property
    def value(self):
        """Returns the most recent value."""
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            value=self.value
        )

# -----------------------------------------------------------------------------
# Main Training Function
# -----------------------------------------------------------------------------
def main():

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. Datasets / Loaders
    # Assuming 'load_dset' is imported from your data module
    train_ds, test_ds = load_dset(args)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
    )

    # 3. Model Initialization
    model = RatTransformer(
        d_model=args.d_model,
        nhead=args.n_head,
        num_layers=args.num_layers,
        n_modes=args.n_modes,
        hist_len=args.hist_len,
        future_len=args.future_len,
        independent_modes=args.independent_modes
    ).to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        betas=(args.beta1, args.beta2)
    )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(args.epochs), eta_min=float(args.eta_min)
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {n_params} trainable parameters.")

    # 4. WandB Logging Setup
    # Define checkpoint path (assuming 'ckpt_save_path' and 'time_str' are global or defined)
    # For this snippet, I'll define a dummy path if not present
    global ckpt_save_path, time_str
    if 'ckpt_save_path' not in globals():
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_save_path = f"./checkpoints/mouse_pred_{time_str}"
    
    if args.debug:
        ckpt_save_path = ckpt_save_path.replace("sequence_ckpt", "sequence_ckpt/debug")
        os.makedirs(ckpt_save_path, exist_ok=True)
        logger.info("Debug mode: WandB logging disabled.")
    else:
        os.makedirs(ckpt_save_path, exist_ok=True)
        wandb.init(
            project="mouse_multimodal_prediction",
            config=vars(args),
            name=f"mouse_multimodal_pred_{time_str}",
        )

    # 5. Training Loop
    best_val = float("inf")
    best_path = os.path.join(ckpt_save_path, "best.pt")
    last_path = os.path.join(ckpt_save_path, "last.pt")

    global_step = 0  # To track total iterations across epochs for WandB
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        
        # metric_logger handles smoothed averaging (window_size=10)
        metric_logger = defaultdict(lambda: SmoothedValue(window_size=10))
        header = f'Epoch: [{epoch}]'

        for batch_idx, batch in enumerate(train_loader):
            hist = batch["hist"].to(device)
            fut = batch["future"].to(device)

            # Generate masks
            hist_mask = make_hist_mask(hist)
            fut_mask = make_future_mask(fut)

            # Handle NaNs
            hist = fill_nan(hist, 0.0)
            fut = fill_nan(fut, 0.0)

            # Compatibility check: if data has extra keypoint dimension, average it 
            # (Assuming you want to reduce to single point if model expects 2 dim)
            if hist.dim() == 4:
                hist = hist.mean(dim=-1)
            if fut.dim() == 4:
                fut = fut.mean(dim=-1)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            conf_logits, mean, logvar = model(hist, hist_mask)
            
            # Compute loss
            loss_dict = wayformer_loss(
                conf_logits, mean, logvar, fut, valid=fut_mask
            )
            loss = loss_dict["loss"]
            loss.backward()

            # Gradient Clipping
            if hasattr(args, "grad_clip") and args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.grad_clip
                )

            optimizer.step()

            # Update metrics using SmoothedValue
            batch_sz = hist.size(0)
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    metric_logger[k].update(v.item(), batch_sz)
                else:
                    metric_logger[k].update(v, batch_sz)
            
            global_step += 1

            # --- Periodic Logging (every N iterations) ---
            if batch_idx % args.print_freq == 0:
                # Construct log message for console
                log_msg = [
                    f"{header}",
                    f"Iter: [{batch_idx}/{len(train_loader)}]",
                    f"lr: {optimizer.param_groups[0]['lr']:.6f}",
                    f"loss: {metric_logger['loss'].avg:.4f}",
                    f"nll: {metric_logger['nll_best'].avg:.4f}",
                    f"l2: {metric_logger['l2_best'].avg:.4f}"
                ]
                logger.info("  ".join(log_msg))

                # Log to WandB (using global_step for smooth x-axis)
                if not args.debug:
                    wandb_logs = {f"train/{k}": v.avg for k, v in metric_logger.items()}
                    wandb_logs['lr'] = optimizer.param_groups[0]['lr']
                    wandb_logs['epoch'] = epoch
                    wandb.log(wandb_logs, step=global_step)

        # Step the scheduler at the end of epoch
        scheduler.step()

        # 6. Validation Step with Timer
        t_val_start = time.time()
        val_logs = evaluate(model, test_loader, device)
        val_duration = time.time() - t_val_start
        
        # Construct Epoch Summary Log
        epoch_logs = {
            "epoch": epoch,
            "val_time": str(datetime.timedelta(seconds=int(val_duration))),
            "val_loss": val_logs.get("val_loss", 0.0)
        }
        
        # Add global averages of training metrics for the entire epoch
        for k, v in metric_logger.items():
            epoch_logs[f"train_{k}_epoch_avg"] = v.global_avg
        
        # Merge val_logs into epoch_logs
        epoch_logs.update(val_logs)

        logger.info(f"End of Epoch {epoch} Summary: {json.dumps(epoch_logs, ensure_ascii=False)}")
        
        # Log validation metrics to WandB
        if not args.debug:
            wandb_val_logs = {f"val/{k}": v for k, v in val_logs.items()}
            # Optionally record validation time
            wandb_val_logs["val/duration_sec"] = val_duration
            wandb.log(wandb_val_logs, step=global_step)

        # 7. Checkpointing
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
            "global_step": global_step,
        }
        
        # Save last checkpoint
        torch.save(state, last_path)
        
        # Save best checkpoint
        if val_logs["val_loss"] < best_val:
            best_val = val_logs["val_loss"]
            torch.save(state, best_path)
            logger.info(
                f"New best checkpoint saved to: {best_path} (val_loss={best_val:.6f})"
            )
        
        # Save periodic checkpoint (every 100 epochs)
        if epoch % args.save_every_epochs == 0:
            epoch_path = os.path.join(ckpt_save_path, f"epoch_{epoch}.pt")
            torch.save(state, epoch_path)
            logger.info(f"Epoch {epoch} checkpoint saved to: {epoch_path}")

    # End of Training
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(
        f"Training finished. Total time: {total_time_str}. Best val_loss={best_val:.6f}. Last checkpoint: {last_path}"
    )

if __name__ == "__main__":
    main()
