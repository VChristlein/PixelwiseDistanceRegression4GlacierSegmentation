#!/bin/bash
#SBATCH --job-name=Glacier
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12000
#SBATCH --gres=gpu:1
#SBATCH -o logfile3.out
#SBATCH -e logfile3.err

 

 
# Small Python packages can be installed in own home directory. Not recommended for big packages like tensorflow -> Follow instructions for pipenv below
# cluster_requirements.txt is a text file listing the required pip packages (one package per line)
#  pip3 install --user -r cluster_requirements.txt
#pip3 install pathlib
#pip3 install scikit-image
python3 Runner3.py