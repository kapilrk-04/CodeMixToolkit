#!/bin/bash
#SBATCH -A nlp
#SBATCH -n 20
#SBATCH --mem-per-cpu=3G
#SBATCH --gres=gpu:2
#SBATCH --time=4-00:00:00
#SBATCH --output=en_enhi_indicbart_train_output.txt

module load u18/cuda/11.7

mkdir -p /scratch/prashantk
python indicbart_model_train.py


