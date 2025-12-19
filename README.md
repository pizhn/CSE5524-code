# Environment Setup

<!-- (swirl) [ping69852@a0208] ~/ps/CSE5524-code$ pip freeze >> requirements.txt
(swirl) [ping69852@a0208] ~/ps/CSE5524-code$ python --version
Python 3.10.19 -->

We recommend setting up a virtual environment using `conda`. Below are the steps to create a virtual environment and install the required packages.

```bash
# Create a new conda environment
conda create -n cse5524_env python=3.10 -y

# Activate the environment
conda activate cse5524_env

# Install required packages
pip install -r requirements.txt
```

```bash

# Train

```bash
# Use the following commands to train the model.

## Baseline algorithm: 1 mode
python train.py --config "config/train/accumulative/6004874/20230117/train_config_1mode.yaml"

## Advanced algortihm: n modes (n=8)

### Normal Training / Rat 6004874 / First day:
python train.py --config "config/train/normal/6004874/20230117/train_config.yaml"

### Normal Training / Rat 6004874 / Second day:
python train.py --config "config/train/normal/6004874/20230117/train_config.yaml"

#### ...

### Accumulative Training / Rat 6004874 / First day:
python train.py --config "config/train/accumulative/6004874/20230121/train_config.yaml"

### Accumulative Training / Rat 6004874 / Second day:
python train.py --config "config/train/accumulative/6004874/20230121/train_config.yaml"

```

# Test

```bash
# After training, use the following commands to test the model.
## Existing model / Test on rat 6004874 / First day:

python test.py --ckp_file "sequence_ckpt/normal/6004874/20230118/20251212-033659/last.pt" --test_config "config/data/normal/6004874/20230117/data.yaml" --test_appendix "first_day"

## Existing model / Test on rat 6004874 / Second day:

python test.py --ckp_file "sequence_ckpt/normal/6004874/20230118/20251212-033659/last.pt" --test_config "config/data/normal/6004874/20230117/data.yaml" --test_appendix "second_day"

```

# Data

Dataset is unavailable due to privacy concerns. Please contact the authors for access to the data.
