#! /usr/bin/env python
import shlex
import argparse
import subprocess
from pathlib import Path

src_dir = Path(__file__).parent.absolute().parent.absolute() / 'src'
log_dir = Path(__file__).parent.absolute().parent.absolute() / 'logs' / 'astro_disposal'
USE_SHELL = False

# DQN params
N_AGENTS = 2
ARCHITECTURE = "v4"
BUFFER = 200000
GAMMA = 0.9
TENSORBOARD_DATA = [str(log_dir), 50, 25, '.log']
USE_DUELING = True
USE_DDQN = True
USE_CNN = True
USE_TENSORBOARD = True

# Train params
N_ITERATIONS = 20000
# N_ITERATIONS = 10
BATCH_SIZE = 64
TRAIN_FREQ = 1
TARGET_FREQ = 10
# ALPHA = 0.003566448247686571
ALPHA = 0.001
TAU = 0.1
INIT_EPS = 1.0
FINAL_EPS = 0.05
EPS_DECAY = 0.75	# for linear eps
# EPS_DECAY = 0.025	# for log eps
CYCLE_EPS = 0.97
EPS_TYPE = "linear"
USE_GPU = True
DEBUG = False
USE_RENDER = False
PRECOMP_FRAC = 0.33

# Environment params
# GAME_LEVEL = ['level_one', 'level_two']
GAME_LEVEL = ['cramped_room']
STEPS_EPISODE = 400
WARMUP_STEPS = STEPS_EPISODE * 2
FIELD_LENGTH = 15
SLIP = False
FACING = True
AGENT_CENTERED = False
USE_ENCODING = True
VERSION = 2

parser = argparse.ArgumentParser()
parser.add_argument('--data-logs', dest='data_logs', type=str, required=False, default=TENSORBOARD_DATA[0])
parser.add_argument('--iterations', dest='max_iterations', type=int, required=False, default=N_ITERATIONS)
parser.add_argument('--eps-type', dest='eps_type', type=str, required=False, default=EPS_TYPE)
parser.add_argument('--eps-decay', dest='eps_decay', type=float, required=False, default=EPS_DECAY)
parser.add_argument('--models-dir', dest='models_dir', type=str, default='', help='Directory to store trained models, if left blank stored in default location')
parser.add_argument('--logs-dir', dest='logs_dir', type=str, default='', help='Directory to store logs, if left blank stored in default location')
parser.add_argument('--restart', dest='restart', action='store_true', help='Flag that signals that train is suppose to restart from a previously saved point.')
parser.add_argument('--checkpoint-file', dest='chkpt_file', type=str, default='', help='Checkpoint file location to use to restart train')
parser.add_argument('--buffer-smart-add', dest='buffer_smart_add', action='store_true',
					help='Flag denoting the use of smart sample add to experience replay buffer instead of first-in first-out')
parser.add_argument('--buffer-method', dest='buffer_method', type=str, required=False, default='uniform', choices=['uniform', 'weighted'],
					help='Method of deciding how to add new experience samples when replay buffer is full')
parser.add_argument('--only-movement', dest='only_movement', action='store_true', help='Flag denoting the training of only moving in the environment')
parser.add_argument('--curriculum-learning', dest='curriculum_learning', action='store_true', help='Flag denoting the use of curriculum learning.')
parser.add_argument('--curriculum-model-path', dest='curriculum_model', type=str, required=False, default='', help='Path to the model to use in curriculum learning.')
parser.add_argument('--initial-temp', dest='init_temp', type=float, default=1.0, help='Initial value for the annealing temperature.')
parser.add_argument('--buffer-size', dest='buffer_size', type=int, required=False, default=BUFFER)


input_args = parser.parse_args()
n_iterations = input_args.max_iterations
eps_type = input_args.eps_type
eps_decay = input_args.eps_decay
data_logs = input_args.data_logs
models_dir = input_args.models_dir
logs_dir = input_args.logs_dir
restart = input_args.restart
chkpt_file = input_args.chkpt_file if input_args.chkpt_file != '' else ''
smart_add = input_args.buffer_smart_add
add_method = input_args.buffer_method
train_only_movement = input_args.only_movement
use_curriculum_learning = input_args.curriculum_learning
curriculum_path = input_args.curriculum_model
anneal_init = input_args.init_temp
buffer_size = input_args.buffer_size


args = (" --nagents %d --architecture %s --buffer %d --gamma %f --iterations %d --batch %d --train-freq %d "
		"--target-freq %d --alpha %f --tau %f --init-eps %f --final-eps %f --eps-decay %f --eps-type %s --warmup-steps %d --cycle-eps-decay %f "
		"--game-levels %s --max-env-steps %d --field-size %d %d --version %d "
		"--tensorboardDetails %s %d %d %s"
		% (N_AGENTS, ARCHITECTURE, buffer_size, GAMMA,  																								# DQN parameters
		   n_iterations, BATCH_SIZE, TRAIN_FREQ, TARGET_FREQ, ALPHA, TAU, INIT_EPS, FINAL_EPS, eps_decay, eps_type, WARMUP_STEPS, CYCLE_EPS,  	# Train parameters
		   ' '.join(GAME_LEVEL), STEPS_EPISODE, FIELD_LENGTH, FIELD_LENGTH, VERSION,  															# Environment parameters
		   data_logs, TENSORBOARD_DATA[1], TENSORBOARD_DATA[2], TENSORBOARD_DATA[3]))
args += ((" --dueling" if USE_DUELING else "") + (" --ddqn" if USE_DDQN else "") + (" --render" if USE_RENDER else "") + ("  --gpu" if USE_GPU else "") +
		 (" --cnn" if USE_CNN else "") + (" --tensorboard" if USE_TENSORBOARD else "") + (" --layer-obs" if USE_CNN else "") +
		 (" --restart --checkpoint-file %s" % chkpt_file if restart else "") + (" --use-curriculum --curriculum-model %s" % curriculum_path if use_curriculum_learning else "") +
		 (" --debug" if DEBUG else "") + (" --has-slip" if SLIP else "") + (" --require_facing" if FACING else "") +
		 (" --agent-centered" if AGENT_CENTERED else "") + (" --use-encoding" if USE_ENCODING else "") + (" --fraction %f" % PRECOMP_FRAC) +
		 (" --models-dir %s" % models_dir if models_dir != '' else "") + (" --logs-dir %s" % logs_dir if logs_dir != '' else "") + (" --buffer-smart-add" if smart_add else "") +
		 (" --buffer-method %s" % add_method) + (" --train-only-movement" if train_only_movement else "") + (" --initial-temp %f" % anneal_init))
commamd = "python " + str(src_dir / 'train_toxic_central_model_dqn.py') + args
if not USE_SHELL:
	commamd = shlex.split(commamd)

print(commamd)
try:
	subprocess.run(commamd, shell=USE_SHELL, check=True)

except subprocess.CalledProcessError as e:
	print(e.output)
