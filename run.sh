#!/bin/bash
SEEDS=(0 42 123 2279 7931)

# Water
for seed in ${SEEDS[@]}; do
    sbatch slurm_launcher.slrm main.py --molecule "H2O"  --optimizer "lbfgs" --ortho "qr" --seed $seed
    sbatch slurm_launcher.slrm main.py --molecule "H2O"  --optimizer "lbfgs" --ortho "cayley" --seed $seed
    sbatch slurm_launcher.slrm main.py --molecule "H2O"  --optimizer "adam" --ortho "qr" --seed $seed
    sbatch slurm_launcher.slrm main.py --molecule "H2O"  --optimizer "adam" --ortho "cayley" --seed $seed
done

# Benzene
for seed in ${SEEDS[@]}; do
    sbatch slurm_launcher.slrm main.py --molecule "C6H6"  --optimizer "lbfgs" --ortho "qr" --seed $seed
    sbatch slurm_launcher.slrm main.py --molecule "C6H6"  --optimizer "lbfgs" --ortho "cayley" --seed $seed
    sbatch slurm_launcher.slrm main.py --molecule "C6H6"  --optimizer "adam" --ortho "qr" --seed $seed
    sbatch slurm_launcher.slrm main.py --molecule "C6H6"  --optimizer "adam" --ortho "cayley" --seed $seed
done

# Graphene
for seed in ${SEEDS[@]}; do
    sbatch slurm_launcher.slrm main.py --molecule "graphene"  --optimizer "lbfgs" --ortho "qr" --seed $seed
    sbatch slurm_launcher.slrm main.py --molecule "graphene"  --optimizer "lbfgs" --ortho "cayley" --seed $seed
    sbatch slurm_launcher.slrm main.py --molecule "graphene"  --optimizer "adam" --ortho "qr" --seed $seed
    sbatch slurm_launcher.slrm main.py --molecule "graphene"  --optimizer "adam" --ortho "cayley" --seed $seed
done