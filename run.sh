#!/bin/bash

sbatch slurm_launcher.slrm main.py \
  --molecule "C6H6" \
  --method "lda" \
  --basis "def2-SVP" \
  --optimizer "Adam" \
  --num_iter 200 \
  --lr 0.01
