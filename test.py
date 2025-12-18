import os
import json
import math
import pickle
import argparse
from datetime import datetime
import logging
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader

import wandb
import yaml

from collections import defaultdict

from common import RatWindowDataset, load_dset, make_hist_mask, make_future_mask, fill_nan, evaluate, wayformer_loss

from scipy.signal import find_peaks

import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# -----------------------------------------------------------------------------
# Use the new Wayformer-style decoder model (learnable query/anchors)
# -----------------------------------------------------------------------------
from model import RatTransformer   # make sure this points to your new model

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--ckp_file", type=str, required=True, help="Checkpoint file name")
parser.add_argument("--test_config", type=str, required=True, help="Path to test config YAML file")
parser.add_argument("--test_appendix", type=str, default="", help="Optional string to append to test log dir")
args = parser.parse_args()

ckp_file = args.ckp_file
ckp_path = os.path.dirname(ckp_file)
config_json_file = os.path.join(ckp_path, "args.json")

with open(config_json_file, "r") as f:
    train_args = json.load(f)

with open(args.test_config, "r") as f:
    test_config = yaml.safe_load(f)

# -----------------------------------------------------------------------------
# Logging / checkpoint paths
# -----------------------------------------------------------------------------
time_str = datetime.now().strftime("%Y%m%d-%H%M%S")

log_save_path = os.path.join(ckp_path, f"test_{args.test_appendix}_{time_str}")
os.makedirs(log_save_path, exist_ok=True)
log_file = os.path.join(log_save_path, "log.txt")

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
logger.info(f"Checkpoint directory: {log_save_path}")


# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------

plot_y_limit = (-80000, 2000)

peak_prominence = 5000.0
peak_distance = 20

# Load train model checkpoint
logger.info(f"Loading model checkpoint from: {ckp_file}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RatTransformer(
        d_model=train_args["d_model"],
        nhead=train_args["n_head"],
        num_layers=train_args["num_layers"],
        n_modes=train_args["n_modes"],
        hist_len=train_args["hist_len"],
        future_len=train_args["future_len"],
        independent_modes=train_args["independent_modes"]
    ).to(device)
model.load_state_dict(torch.load(ckp_file, map_location=device)["model"])
model.eval()

idx_target = train_args["idx_target"]
hist_len = train_args["hist_len"]
future_len = train_args["future_len"]
downsample_factor = train_args["downsample_factor"]

all_segment_ds = [] 
all_time_ds = []

for pkl_file in test_config["data_files"]:
    print(f"Processing test pkl: {pkl_file}")
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    segment = data["segment"]
    time = data["time"]
    time_zeroed = time - time[0]
    assert np.all(time_zeroed[1:] - time_zeroed[:-1] >= 0), "Time values must be strictly increasing."
    # Select target rat index and downsample in time
    segment_ds = segment[::downsample_factor][:, :, :, idx_target]
    time_ds = time_zeroed[::downsample_factor]

    assert segment_ds.shape[1] == 1, "Expected number of rats = 1 in dim=1."
    segment_ds = segment_ds.squeeze(1)
    all_segment_ds.append(segment_ds)
    all_time_ds.append(time_ds)    

test_ds = RatWindowDataset(all_segment_ds, hist_len, future_len, data_time_list=all_time_ds, return_idx=True, file_names=test_config["data_files"])
test_loader = DataLoader(
    test_ds,
    batch_size=1024,
    shuffle=False,
    drop_last=False,
)

all_extractions = defaultdict(dict)

extraction_save_path = os.path.join(ckp_path, f"extractions_{args.test_appendix}.pkl")

if os.path.exists(extraction_save_path):
    logger.info(f"Loading existing extractions from {extraction_save_path}")
    with open(extraction_save_path, "rb") as f:
        all_extractions = pickle.load(f)
else:
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating test dataset"):
            hist = batch["hist"].to(device=device, dtype=torch.float32)
            fut = batch["future"].to(device=device, dtype=torch.float32)

            hist_mask = make_hist_mask(hist)  # [1, Th]
            fut_mask = make_future_mask(fut)  # [1, H]

            hist = fill_nan(hist, 0.0)
            fut = fill_nan(fut, 0.0)

            if hist.dim() == 4:
                assert hist.shape[-1] == 1
                hist = hist.mean(dim=-1)
            if fut.dim() == 4:
                assert fut.shape[-1] == 1
                fut = fut.mean(dim=-1)

            # hist, fut, conf_logits, mean, logvar

            conf_logits, mean, logvar = model(hist, hist_mask)  # mean/logvar: [1,K,H,2]
            loss_dict = wayformer_loss(conf_logits, mean, logvar, fut, valid=fut_mask)

            ll = -loss_dict["nll_per_data"].cpu().numpy()
            hist_start, hist_end, fut_start, fut_end = batch["time_interval"]
            # file_index = batch["file_index"]
            file_names = batch["file_name"]

            for _ll, _file_name, _hs, _he, _fs, _fe, _hist, _fut, _conf_logits, _mean, _logvar in zip(
                ll,
                file_names,
                hist_start,
                hist_end,
                fut_start,
                fut_end,
                hist,
                fut,
                conf_logits,
                mean,
                logvar
            ):
                _ll = float(_ll)
                _hs = float(_hs)
                _he = float(_he)
                _fs = float(_fs)
                _fe = float(_fe)
                # ##################################################
                _file_name = "_".join(_file_name.split("/")[-2:]).replace(".pkl", "")
                # ##################################################
                # all_lls[_file_name][(_hs, _he, _fs, _fe)] = _ll
                all_extractions[_file_name][(_hs, _he, _fs, _fe)] = {
                    "log_likelihood": _ll,
                    "hist": _hist.cpu().numpy(),
                    "future": _fut.cpu().numpy(),
                    "conf_logits": _conf_logits.cpu().numpy(),
                    "mean": _mean.cpu().numpy(),
                    "logvar": _logvar.cpu().numpy()
                }
    logger.info(f"Saving extractions to {extraction_save_path}")
    with open(extraction_save_path, "wb") as f:
        pickle.dump(all_extractions, f)

for file_name in all_extractions:
    plot_file = os.path.join(log_save_path, f"loss_{file_name}.png")

    all_x = []
    all_ll = []
    all_hist = []
    all_future = []
    all_pred = []

    for (_hs, _he, _fs, _fe), extraction in all_extractions[file_name].items():
        _ll = extraction["log_likelihood"]
        _hist = extraction["hist"]
        _fut = extraction["future"]
        _mean = extraction["mean"]
        _conf_logits = extraction["conf_logits"]
        _conf_ind = np.argmax(_conf_logits, axis=0)
        _pred = _mean[_conf_ind]
        all_x.append((_he + _fs) / 2.0)
        all_ll.append(_ll)
        all_hist.append(_hist)
        all_future.append(_fut)
        all_pred.append(_pred)

    all_x = np.array(all_x)
    all_ll = np.array(all_ll)
    all_hist = np.array(all_hist)
    all_future = np.array(all_future)
    all_pred = np.array(all_pred)

    outlier_indices, _ = find_peaks(-all_ll, prominence=peak_prominence, distance=peak_distance)
    outlier_x = all_x[outlier_indices]
    outlier_ll = all_ll[outlier_indices]

    # Export outliers to a file
    outlier_x = all_x[outlier_indices]
    outlier_ll = all_ll[outlier_indices]
    outlier_hist = all_hist[outlier_indices]
    outlier_future = all_future[outlier_indices]
    outlier_pred = all_pred[outlier_indices]
    outlier_save_file = os.path.join(log_save_path, f"outliers_{file_name}.pkl")

    with open(outlier_save_file, "wb") as f:
        pickle.dump({
            "all_x": all_x,
            "all_ll": all_ll,
            "all_hist": all_hist,
            "all_future": all_future,
            "all_pred": all_pred,
            "x": outlier_x,
            "ll": outlier_ll,
            "hist": outlier_hist,
            "future": outlier_future,
            "pred": outlier_pred
        }, f)

    plt.figure(figsize=(10, 6))
    plt.title(f"Log-Likelihoods for {file_name}")
    plt.xlabel("Time")
    plt.ylabel("Log-Likelihood")
    plt.ylim(plot_y_limit)

    plt.plot(
        all_x, 
        all_ll, 
        marker='o', 
        linestyle='-', 
        linewidth=1, 
        markersize=2,
        label='Log-Likelihood'
    )
    if len(outlier_indices) > 0:
        plt.scatter(outlier_x, outlier_ll, color='red', marker='x', s=120, label='Outliers')

    # plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
