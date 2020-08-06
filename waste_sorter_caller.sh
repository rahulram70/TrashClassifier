#!/bin/bash

#SBATCH --partition=exacloud
#SBATCH --mem=10000
#SBATCH --time=2159

srun python waste_sorter.py
