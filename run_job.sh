#!/bin/bash
#SBATCH --job-name=ThesisRun
#SBATCH --account=project_2015651
#SBATCH --partition=small
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --mail-type=BEGIN

source /scratch/project_2015651/Masters_thesis/venv/bin/activate
cd /scratch/project_2015651/Masters_thesis/src

python main.py

