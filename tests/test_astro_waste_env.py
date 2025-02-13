#! /usr/bin/env python

import numpy as np

from env.toxic_waste_env_base import PlayerState
from env.toxic_waste_env_v2 import WasteStateV2, ToxicWasteEnvV2, Actions, ProblemType, WasteType
from env.astro_greedy_agent import GreedyAgent
from itertools import permutations
from pathlib import Path
from typing import List, Union, Dict
import jax

from algos.dqn import DQNetwork
import flax.linen as nn
import yaml
import argparse
import logging
from datetime import datetime

RNG_SEED = 18102023
N_CYCLES = 250
ACTION_DIM = 6
def get_model_obs(raw_obs: Union[np.ndarray, Dict]) -> np.ndarray:
	if isinstance(raw_obs, dict):
		model_obs = np.array([raw_obs['conv'].reshape(1, *raw_obs['conv'].shape), np.array(raw_obs['array'])],
							 dtype=object)
	else:
		model_obs = np.array([raw_obs[0].reshape(1, *raw_obs[0].shape), raw_obs[1:]], dtype=object)
	
	return model_obs

parser = argparse.ArgumentParser(description='Train DQN model for Astro waste disposal game.')

parser.add_argument('--buffer', dest='buffer_size', type=int, required=True, help='Size of the replay buffer in the DQN')
parser.add_argument('--gamma', dest='gamma', type=float, required=False, default=0.99, help='Discount factor for agent\'s future rewards')
parser.add_argument('--gpu', dest='use_gpu', action='store_true', help='Flag that signals the use of gpu for the training')
parser.add_argument('--tensorboard', dest='use_tensorboard', action='store_true',
					help='Flag the signals the use of a tensorboard summary writer. Expects argument --tensorboardDetails to be present')
parser.add_argument('--tensorboardDetails', dest='tensorboard_details', nargs='+', required=False, default=None,
					help='List with the details for the tensorboard summary writer: <log_dirname: str>, <queue_size :int>, <flush_time: int>, <suffix: str>'
							' Use only in combination with --tensorboard option')
args = parser.parse_args()

buffer_size = args.buffer_size
gamma = args.gamma
use_gpu = args.use_gpu
use_tensorboard = args.use_tensorboard
tensorboard_details = args.tensorboard_details

home_dir = Path(__file__).parent.absolute().parent.absolute()
models_dir = home_dir / 'models'/ 'best' / 'only_movement'
astro_model_filename = 'cramped_room_move_vdn_astro_model.model'
human_model_filename = 'cramped_room_move_vdn_human_model.model'

def main():
	
	field_size = (15, 15)
	layout = 'cramped_room'
	n_players = 2
	has_slip = False
	n_objects = 3
	max_episode_steps = 500
	facing = True
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
	problem_type = "move_catch"
	env = ToxicWasteEnvV2(field_size, layout, n_players, n_objects, max_episode_steps, RNG_SEED, data_dir, facing, centered_obs,
	                      slip=has_slip, is_train=True, use_render=use_render, pick_all=False, problem_type=ProblemType.ONLY_MOVE)
	now = datetime.now()
	log_filename = ('test_astro_waste_env_%s' % layout + '_' + now.strftime("%Y%m%d-%H%M%S"))
	log_dir = home_dir / 'logs'
	logger = logging.getLogger("%s" % layout)
	logger.setLevel(logging.INFO)
	file_handler = logging.FileHandler(log_dir / (log_filename + '.log'))
	file_handler.setFormatter(logging.Formatter('%(name)s %(asctime)s %(levelname)s:\t%(message)s'))
	file_handler.setLevel(logging.INFO)
	logger.addHandler(file_handler)
	
	env.render()
	obs, *_ = env.reset(seed=RNG_SEED)
	model_obs = get_model_obs(obs[0])
	obs_shape = (model_obs[0].shape ,model_obs[1].shape)
	print(obs_shape)
	action_dim = env.action_space[0].n
	astro_dqn = DQNetwork(action_dim, n_layers, nn.relu, layer_sizes, gamma, cnn_layer=True, dueling_dqn=True, use_ddqn=True, cnn_properties=cnn_properties)
	print("params agent:", action_dim, n_layers, nn.relu, layer_sizes, gamma, True, True, True, True, cnn_properties)

	astro_dqn.load_model_v2(astro_model_filename, models_dir, logger, obs_shape)
	human_dqn = DQNetwork(action_dim, n_layers, nn.relu, layer_sizes, gamma, cnn_layer=True,dueling_dqn=True, use_ddqn=True, cnn_properties=cnn_properties)
	print("params agent2:", action_dim, n_layers, nn.relu, layer_sizes, gamma, True, True, True, True, cnn_properties)
	
	human_dqn.load_model_v2(human_model_filename, models_dir, logger, obs_shape)
	
	successes = 0
	agent_models.append(astro_dqn)
	agent_models.append(human_dqn)
	state, *_ = env.reset(seed=RNG_SEED)
	env.seed(RNG_SEED)
	np.random.seed(RNG_SEED)
	rng_gen = np.random.default_rng(RNG_SEED)
	agent_ids = [agent.name for agent in env.players]
	n_actions = env.action_space[0].n
	for i in range(N_CYCLES):
		obs, *_ = env.reset(seed=RNG_SEED)
		done = False
		print('Iteration: %d' % (i + 1))
		epoch = 0
		while not done:
			print('Epoch %d' % (epoch + 1))
			actions = []
			for idx in range(n_players):
				dqn_model = agent_models[idx]
				model_obs = get_model_obs(obs[idx])
				q_values = dqn_model.q_network.apply(dqn_model.online_state.params, model_obs[0], model_obs[1])[0]
				pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
				pol = pol / pol.sum()
				action = rng_gen.choice(range(n_actions), p=pol)
				
				actions += [int(jax.device_get(action))]
				print('Player %s at (%d, %d) with orientation (%d, %d) and chose action %s' % (env.players[idx].name, *env.players[idx].position,
				 																							*env.players[idx].orientation, Actions(action).name))
				print('Wastes: ', str(env.objects))
	
			next_obs, rewards, finished, timeout, info = env.step(actions)
			obs = next_obs
			env.render()
			epoch += 1
			print(finished, timeout)
			# print(env.get_filled_field())
			if finished or timeout:
				done = True
				env.reset()
				if finished:
					successes += 1
				input()

	print('Finished %f of attempts' % (successes / N_CYCLES))
	



if __name__ == '__main__':
	main()
