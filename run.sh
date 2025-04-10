#!/bin/bash
SEEDS=(0 42 123 2279 7931 12345 54321 99999 100000 100001)
OPTIMIZERS=("adam" "bfgs") # "radam") # "rbfgs") #
ORTHOGONALIZERS=("cayley" "qr" "polar" "matexp")
MOLECULES=("CH" "OH" "NiCH2+" "CoCO+" "NiCO+" "ScCO+" "Fe(CO)2+")

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
