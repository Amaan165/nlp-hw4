#!/bin/bash
#SBATCH --job-name=hw4_bert
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=n1s8-v100-1
#SBATCH --output=slurm_%j.out

# Load modules
module purge
module load anaconda3/2020.07

# Activate environment
source activate hw4-part-1-nlp

# for debugging first:
python main.py --train --eval --debug_train

# Run training
# python main.py --train --eval

