#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=miguel.faria@tecnico.ulisboa.pt
#SBATCH --job-name=train_toxic_central_model
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=shard:10
#SBATCH --time=144:00:00
#SBATCH --mem-per-cpu=1000
date;hostname;pwd

source $HOME/python_envs/deep_rl_env/bin/activate
python "$HOME"/Documents/Projects/Toxic_Waste_Gym/scripts/run_train_toxic_central_model.py

deactivate
date

