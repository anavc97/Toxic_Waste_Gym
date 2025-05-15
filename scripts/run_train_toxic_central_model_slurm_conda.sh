#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ana.vilaca.c@tecnico.ulisboa.pt
#SBATCH --job-name=train_toxic_central_model
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --mem=50G
#SBATCH --qos=gpu-medium
#SBATCH --output=/home/users/acarrasco/projects/toxic_waste/logs/"job-%x-%j.out"
date;hostname;pwd

if [ -n "${SLURM_JOB_ID:-}" ] ; then
  script_path=$(dirname "$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}' | head -n 1)")
else
  script_path="$( cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )"
fi

export LD_LIBRARY_PATH="/usr/lib/cuda/lib64:$LD_LIBRARY_PATH"
export PATH="/usr/lib/cuda/bin:$PATH"

if [ "$HOSTNAME" = "artemis" ] || [ "$HOSTNAME" = "poseidon" ] ; then
  source "$HOME"/miniconda3/bin/activate deep_rl_env
  python "$script_path"/run_train_toxic_central_model.py --data-logs /mnt/scratch-artemis/miguelfaria/logs/toxic_waste --logs-dir /mnt/scratch-artemis/miguelfaria/logs/toxic_waste --models-dir /mnt/data-artemis/miguelfaria/toxic_waste --buffer-method uniform --initial-temp 0.0 --only-movement --iterations 6000 --eps-type linear --eps-decay 0.5 --buffer-size 10000 # --restart --checkpoint-file "$chkpt_dir"/v2_train_checkpoint_data.json
elif [ "$HOSTNAME" = "nexus1" ] || [ "$HOSTNAME" = "nexus2" ] || [ "$HOSTNAME" = "nexus3" ] || [ "$HOSTNAME" = "nexus4" ]; then
  source "$HOME"/python_envs/toxic_waste_env/bin/activate
  python "$script_path"/run_train_toxic_central_model.py --data-logs /home/users/acarrasco/projects/toxic_waste/logs --models-dir /home/users/acarrasco/projects/toxic_waste/models --buffer-method uniform --initial-temp 0.0 --problem-type only_movement --iterations 2010 --eps-type linear --start-eps 0.9 --final-eps 0.1 --eps-decay 0.5 --buffer-size 250000 --checkpoint-freq 1000 --batch-size 64 # --restart --checkpoint-file "$chkpt_dir"/v2_train_checkpoint_data.json

else
  python "$script_path"/run_train_toxic_central_model.py --buffer-method uniform --initial-temp 0.0 --only-movement --iterations 10 --eps-type linear --eps-decay 0.2 --buffer-size 10000
fi

source "$HOME"/miniconda3/bin/deactivate
date

