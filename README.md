# Train

```bash

# Normal Training / Rat 6004874 / First day:
python train.py --config "config/train/normal/6004874/20230117/train_config.yaml"

# Normal Training / Rat 6004874 / Second day:
python train.py --config "config/train/normal/6004874/20230117/train_config.yaml"

# ...

# Accumulative Training / Rat 6004874 / First day:
python train.py --config "config/train/accumulative/6004874/20230121/train_config.yaml"

# Accumulative Training / Rat 6004874 / Second day:
python train.py --config "config/train/accumulative/6004874/20230121/train_config.yaml"

```

# Test

```bash

# Existing model / Test on rat 6004874 / First day:

python test.py --ckp_file "sequence_ckpt/normal/6004874/20230118/20251212-033659/last.pt" --test_config "config/data/normal/6004874/20230117/data.yaml" --test_appendix "first_day"

# Existing model / Test on rat 6004874 / Second day:

python test.py --ckp_file "sequence_ckpt/normal/6004874/20230118/20251212-033659/last.pt" --test_config "config/data/normal/6004874/20230117/data.yaml" --test_appendix "second_day"

```
