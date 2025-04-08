#! /bin/bash

python -u full_experiments/run_bag_of_KANs_experiment.py \
    --data_path datasets/aqueous_solubility_delaney.csv\
    --n_trials 100 \
    --variance_threshold 0.1 \
    --random_seed 1738 \
    --test_size 0.2 \
    --num_itrs 1000 \
    --lr 0.001 \
    --num_bootstraps 25 \
    --parallel