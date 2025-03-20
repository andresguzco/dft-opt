#!/bin/bash
SEEDS=(0 42 123 2279 7931 12345 54321 99999 12531 5234)
OPTIMIZERS=("bfgs" "adam") # "radam") # 
ORTHOGONALIZERS=("cayley" "qr" "polar" "matexp")
MOLECULES=("H2O" "C6H6" "ScCO+")


for seed in ${SEEDS[@]}; do
    for mol in ${MOLECULES[@]}; do
        for opt in ${OPTIMIZERS[@]}; do
            for orth in ${ORTHOGONALIZERS[@]}; do
                sbatch slurm_launcher.slrm main.py \
                --molecule $mol \
                --optimizer $opt \
                --ortho $orth \
                --seed $seed \
                --iters 500 \
                --lr 0.01
            done
        done
    done
done