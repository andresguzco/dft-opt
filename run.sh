#!/bin/bash

sbatch slurm_launcher.slrm main.py --molecule "water" --optimizer "BFGS" --ortho "cayley"
sbatch slurm_launcher.slrm main.py --molecule "water" --optimizer "BFGS" --ortho "qr"
sbatch slurm_launcher.slrm main.py --molecule "water" --optimizer "Adam" --ortho "cayley"
sbatch slurm_launcher.slrm main.py --molecule "water" --optimizer "Adam" --ortho "qr"

sbatch slurm_launcher.slrm main.py --molecule "benzene" --optimizer "BFGS" --ortho "cayley"
sbatch slurm_launcher.slrm main.py --molecule "benzene" --optimizer "BFGS" --ortho "qr"
sbatch slurm_launcher.slrm main.py --molecule "benzene" --optimizer "Adam" --ortho "cayley"
sbatch slurm_launcher.slrm main.py --molecule "benzene" --optimizer "Adam" --ortho "qr"

sbatch slurm_launcher.slrm main.py --molecule "graphene" --optimizer "BFGS" --ortho "cayley"
sbatch slurm_launcher.slrm main.py --molecule "graphene" --optimizer "BFGS" --ortho "qr"
sbatch slurm_launcher.slrm main.py --molecule "graphene" --optimizer "Adam" --ortho "cayley"
sbatch slurm_launcher.slrm main.py --molecule "graphene" --optimizer "Adam" --ortho "qr"