#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=miguel.faria@tecnico.ulisboa.pt
#SBATCH --job-name=train_toxic_multi_model
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=4G
#SBATCH --qos=gpu-medium
#SBATCH --output="job-%x-%j.out"
date;hostname;pwd

if [ -n "${SLURM_JOB_ID:-}" ] ; then
  script_path=$(dirname "$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}' | head -n 1)")
else
  script_path="$( cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )"
fi

export LD_LIBRARY_PATH="/usr/lib/cuda/lib64:$LD_LIBRARY_PATH"
export PATH="/usr/lib/cuda/bin:$PATH"
source "$HOME"/miniconda3/bin/activate deep_rl_env

if [ "$HOSTNAME" = "artemis" ] || [ "$HOSTNAME" = "poseidon" ] ; then
  python "$script_path"/run_train_toxic_multi_model.py --logs-dir /mnt/scratch-artemis/miguelfaria/logs/toxic_waste --models-dir /mnt/data-artemis/miguelfaria/toxic_waste/models --data-dir /mnt/data-artemis/miguelfaria/toxic_waste/data --buffer-method uniform --initial-temp 0.0 --only-movement --iterations 800 --eps-type log --eps-decay 0.1 --buffer-size 1000 --batch-size 32 --game-levels cramped_room level_one level_two # --restart --checkpoint-file "$chkpt_dir"/v2_train_checkpoint_data.json
else
  python "$script_path"/run_train_toxic_multi_model.py --buffer-method uniform --initial-temp 0.0 --only-movement --iterations 10 --eps-type linear --eps-decay 0.2 --buffer-size 10000
fi

source "$HOME"/miniconda3/bin/deactivate
date


