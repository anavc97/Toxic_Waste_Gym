#! /usr/bin/env python

import numpy as np

from env.toxic_waste_env_base import PlayerState
from env.toxic_waste_env_v2 import WasteStateV2, ToxicWasteEnvV2, Actions, ProblemType, WasteType
from env.astro_greedy_agent import GreedyAgent
from itertools import permutations
from pathlib import Path
from typing import List, Union, Dict
import jax
import time
import pickle
from collections import defaultdict
from algos.dqn import DQNetwork
import flax.linen as nn
import yaml
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
import json

RNG_SEED = 18102023
ACTION_DIM = 6
env = None
def get_model_obs(raw_obs: Union[np.ndarray, Dict]) -> np.ndarray:
	if isinstance(raw_obs, dict):
		model_obs = np.array([raw_obs['conv'].reshape(1, *raw_obs['conv'].shape), np.array(raw_obs['array'])],
							 dtype=object)
	else:
		model_obs = np.array([raw_obs[0].reshape(1, *raw_obs[0].shape), raw_obs[1:]], dtype=object)
	
	return model_obs

parser = argparse.ArgumentParser(description='Train DQN model for Astro waste disposal game.')

parser.add_argument('--gamma', dest='gamma', type=float, required=False, default=0.99, help='Discount factor for agent\'s future rewards')
parser.add_argument('--model-name', dest='model_name', type=str, required=True, help='Name of model to load')
parser.add_argument('--problem-type', dest='problem_type', type=str, required=True, help='Problem Type (Folder) for model to load')
parser.add_argument('--iteration', dest='it', type=int, required=False, help='Iteration of model checkpoint to load')
parser.add_argument('--cycles', dest='n_cycles', type=int, required=True, help='How many cycles to run this for')
args = parser.parse_args()

problem_type = args.problem_type
gamma = args.gamma
model_name = args.model_name
N_CYCLES = args.n_cycles
it = args.it

home_dir = Path(__file__).parent.absolute().parent.absolute()

if model_name == "best" or it == None:
	models_dir = home_dir / 'models'/ model_name / problem_type
	astro_model_filename = f'cramped_room_vdn_astro_model.model'
	human_model_filename = f'cramped_room_vdn_human_model.model'	
elif it != None:
	models_dir = home_dir / 'models'/ model_name / problem_type / 'checkpoints'
	astro_model_filename = f'astro-v2_lvl_cramped_room_{problem_type}_it_{it}_checkpoint.model'
	human_model_filename = f'human-v2_lvl_cramped_room_{problem_type}_it_{it}_checkpoint.model'
	
print("LOADING MODEL: ", astro_model_filename)

def get_env_state(env):
	state = ""
	for player in env.players:
		state += f"{player.position[0]}"
		state += f"{player.position[1]}"
	for player in env.players:
		state += f"{player.orientation[0]}"
		state += f"{player.orientation[1]}"
	for obj in env.objects:
		state += f"{obj.position[0]}"
		state += f"{obj.position[1]}"
	for obj in env.objects:
		state += f"{obj.hold_state}"
	for obj in env.objects:
		state += f"{int(obj.identified)}"
		#input(state)
	return str(state)

state_action_table = []
state_action_table.append(defaultdict(lambda: np.zeros(7)))
state_action_table.append(defaultdict(lambda: np.zeros(7)))

def main():
	
	field_size = (15, 15)
	layout = 'cramped_room'
	n_players = 2
	has_slip = False
	n_objects = 3
	max_episode_steps = 500
	facing = False
	layer_obs = True
	centered_obs = True
	use_render = True
	rng_gen = np.random.default_rng(RNG_SEED)
	agent_models = []
	data_dir = data_dir = Path(__file__).parent.absolute().parent.absolute() / 'data'
	configs_dir = data_dir / 'configs'
	architecture = "v3"
	with open(configs_dir / 'q_network_architectures.yaml') as architecture_file:
		arch_data = yaml.safe_load(architecture_file)
		if architecture in arch_data.keys():
			n_layers = arch_data[architecture]['n_layers']
			layer_sizes = arch_data[architecture]['layer_sizes']
			n_conv_layers = arch_data[architecture]['n_cnn_layers']
			cnn_size = arch_data[architecture]['cnn_size']
			cnn_kernel = [tuple(elem) for elem in arch_data[architecture]['cnn_kernel']]
			cnn_strides = arch_data[architecture]['cnn_strides']
			pool_window = [tuple(elem) for elem in arch_data[architecture]['pool_window']]
			pool_strides = arch_data[architecture]['pool_strides']
			pool_padding = [[tuple(dims) for dims in elem] for elem in arch_data[architecture]['pool_padding']]
			cnn_properties = [n_conv_layers, cnn_size, cnn_kernel, cnn_strides, pool_window, pool_strides, pool_padding]
	
	env_version = 2
	env = ToxicWasteEnvV2(field_size, layout, n_players, n_objects, max_episode_steps, RNG_SEED, data_dir, facing, centered_obs,
	                      slip=has_slip, is_train=False, random_init_pos=False, use_render=use_render, pick_all=False, problem_type=problem_type)
	
	eps = 0.0
	eps_decay = 1/(N_CYCLES//1.7)
	now = datetime.now()
	log_filename = ('create_table_model_%s' % model_name + '_' + now.strftime("%Y%m%d-%H%M%S"))
	log_dir = home_dir / 'logs'
	logger = logging.getLogger("%s" % layout)
	logger.setLevel(logging.INFO)
	file_handler = logging.FileHandler(log_dir / (log_filename + '.log'))
	file_handler.setFormatter(logging.Formatter('%(name)s %(asctime)s %(levelname)s:\t%(message)s'))
	file_handler.setLevel(logging.INFO)
	logger.addHandler(file_handler)
	
	obs, *_ = env.reset(seed=RNG_SEED)
	model_obs = get_model_obs(obs[0])	

	obs_shape = (model_obs[0].shape ,model_obs[1].shape)
	action_dim = env.action_space[0].n	
	training = False
	astro_dqn = DQNetwork(action_dim, n_layers, nn.relu, layer_sizes, gamma, training=training, cnn_layer=True, dueling_dqn=True, use_ddqn=True, cnn_properties=cnn_properties)
	astro_dqn.load_model_v2(astro_model_filename, models_dir, logger, obs_shape)
	human_dqn = DQNetwork(action_dim, n_layers, nn.relu, layer_sizes, gamma, training=training, cnn_layer=True,dueling_dqn=True, use_ddqn=True, cnn_properties=cnn_properties)
	
	human_dqn.load_model_v2(human_model_filename, models_dir, logger, obs_shape)
	
	successes = 0
	agent_models.append(human_dqn)
	agent_models.append(astro_dqn)

	state, *_ = env.reset(seed=RNG_SEED)
	if use_render: env.render()
	env.seed(RNG_SEED)
	np.random.seed(RNG_SEED)
	rng_gen = np.random.default_rng(RNG_SEED)
	all_positions=[]
	all_hold_states = [[] for _ in range(env.n_objects)]
	all_object_positions = [[] for _ in range(env.n_objects)]
	n_actions = env.action_space[0].n
	for i in tqdm(range(N_CYCLES)):
		obs, *_ = env.reset()
		done = False
		#print('Iteration: %d' % (i + 1))
		epoch = 0
		stuck = 0
		while not done:
			#print('Epoch %d' % (epoch + 1))
			actions = []
			#logger.info(env.get_env_log())
			state = get_env_state(env)
			old_positions = []				
			if rng_gen.random() < eps:
				for idx in range(n_players):
					old_positions.append(env.players[idx].position)
					actions += [np.random.randint(0,6)]
			else:
				for idx in range(n_players):
					old_positions.append(env.players[idx].position)
					dqn_model = agent_models[idx]
					model_obs = get_model_obs(obs[idx])
					q_values = dqn_model.q_network.apply(dqn_model.online_state.params, model_obs[0], model_obs[1])[0]
					pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
					pol = pol / pol.sum()
					action = rng_gen.choice(range(n_actions), p=pol)
					
					actions += [int(jax.device_get(action))]
					#print('Player %s at (%d, %d) with orientation (%d, %d) and chose action %s' % (env.players[idx].name, *env.players[idx].position,
																												#*env.players[idx].orientation, Actions(action).name))
					print(f"STATE: {state} | ACTION: {action}")
					input()
					state_action_table[idx][state][action] += 1
					if env.players[idx].position not in all_positions: all_positions.append(env.players[idx].position)
					for i,obj in enumerate(env.objects):
						if obj.position not in all_object_positions[i]: all_object_positions[i].append(obj.position) 
						if obj.hold_state not in all_hold_states[i]: all_hold_states[i].append(obj.hold_state)

			next_obs, _, finished, timeout, _ = env.step(actions)

			if use_render: env.render()
			obs = next_obs
			epoch += 1
			if env.players[0].position == old_positions[0] and env.players[1].position == old_positions[1]:
				stuck += 1
				if stuck > 10: done = True
			if finished or timeout:
				done = True
				env.reset()
				if finished:
					successes += 1
		eps -= eps_decay


	for idx in range(n_players):
		for state, counts in state_action_table[idx].items():
			total = sum(counts)
			if total > 0:
				state_action_table[idx][state] = [count / total for count in counts]
			else:
				# If no actions were taken for a state (total == 0), you can decide how to handle it.
				# For example, leave it as zeros or assign a uniform probability.
				state_action_table[idx][state] = counts

		state_action_probs = {str(state): list(probs) for state, probs in state_action_table[idx].items()}

		filename = f"state_action_table_{N_CYCLES}_{env.players[idx].name}.json" 
		with open(filename, "w") as f:
			json.dump(state_action_probs, f)
	
	print('Finished %f of attempts' % (successes / N_CYCLES))
	logger.info('Finished %f of attempts' % (successes / N_CYCLES))

	for idx in range(n_players):
		num_keys = len(state_action_table[idx])
		avg_actions_per_key = np.mean([sum(values) for values in state_action_table[idx].values()])
    
		print(f"Player {idx}:")
		print(f"  Number of unique states (keys): {num_keys}")
		print(f"  Average actions taken per state: {avg_actions_per_key:.2f}")
	
	print(f"  {len(all_positions)} unique positions visited.")

	for i in range(len(env.objects)):
		print(f"Object {i} had {len(all_object_positions[i])} unique positions and {len(all_hold_states[i])} unique hold states.")
	



if __name__ == '__main__':
	main()
