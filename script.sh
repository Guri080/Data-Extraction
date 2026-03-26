#!/bin/bash

#SBATCH -N 1
#SBATCH -c 10
#SBATCH -t 0-24:00:00
#SBATCH -p public
#SBATCH --gres=gpu:a100:1
#SBATCH -q public
#SBATCH -o slurm.run1.out
#SBATCH -e slurm.run1.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user="gssodhi@asu.edu"
#SBATCH --export=NONE
#SBATCH --mem=40G
#SBATCH -A grp_nrolston
#SBATCH -J dataExtract

# Load modules and activate environment
module load mamba/latest
source activate myENV

# Move to project directory
cd /home/gssodhi/comp_vis/Data_Extraction

python main.py