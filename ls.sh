#!/bin/bash
#SBATCH -a 1-1
#SBATCH --job-name KMC
#SBATCH --partition compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=12G
#SBATCH --nodelist=node01

ls -hl /scratch/zhaocj/*/*
