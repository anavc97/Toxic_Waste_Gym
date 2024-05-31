#! /usr/bin/env python

import sys
import argparse
import wandb
import numpy as np
import threading
import jax
import flax.linen as nn
import os
import random
import time
import yaml
import logging

from algos.dqn import EPS_TYPE, DQNetwork
from algos.single_model_madqn import CentralizedMADQN
from env.toxic_waste_env_v1 import ToxicWasteEnvV1
from env.toxic_waste_env_v2 import ToxicWasteEnvV2, Actions, AgentType
from env.astro_greedy_agent import GreedyAgent
from pathlib import Path
from itertools import product
from typing import List, Union, Dict, Tuple, Callable
from datetime import datetime
from itertools import permutations


RNG_SEED = 21062023
ROBOT_NAME = 'astro'
INTERACTIVE_SESSION = False
ANNEAL_TEMP = 0.999


def convert_joint_act(action: int, num_agents: int, n_actions: int) -> List[int]:
	actions_map = list(product(range(n_actions), repeat=num_agents))
	return np.array(actions_map[action])


def get_history_entry(obs: ToxicWasteEnvV2.Observation, actions: List[int], n_agents: int) -> List:
	
	entry = []
	for a_idx in range(n_agents):
		agent = obs.players[a_idx]
		action = actions[a_idx]
		entry += [agent.id, agent.position, agent.orientation, agent.held_objects, action]
	
	return entry


def input_callback(env: Union[ToxicWasteEnvV1, ToxicWasteEnvV2], stop_flag: threading.Event):
	try:
		while not stop_flag.is_set():
			command = input('Interactive commands:\n\trender - display renderization of the interaction\n\tstop_render - stops the renderization\nCommand: ')
			if command == 'render':
				env.use_render = True
			elif command == 'stop_render':
				if env.use_render:
					env.use_render = False
	
	except KeyboardInterrupt as ki:
		return

def model_execution(dqn_model: DQNetwork, eps: float, get_model_obs: Callable, greedy_actions: bool, n_agents: int, n_joint_actions: int,
					obs: np.ndarray, rng_gen: np.random.Generator, waste_env: Union[ToxicWasteEnvV1, ToxicWasteEnvV2], episode_q_vals: List) -> List[int]:
	
	if rng_gen.random() < eps:
		actions = waste_env.action_space.sample()
	else:
		v2_obs = get_model_obs(obs)
		q_values = dqn_model.q_network.apply(dqn_model.online_state.params, v2_obs[0], v2_obs[1].reshape((1, 1)))[0]
		
		if greedy_actions:
			action = q_values.argmax(axis=-1)
		else:
			pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
			pol = pol / pol.sum()
			action = rng_gen.choice(range(n_joint_actions), p=pol)
		joint_action = int(jax.device_get(action))
		actions = convert_joint_act(joint_action, n_agents, waste_env.action_space[0].n)
		episode_q_vals += [float(q_values[int(joint_action)])]
	
	return actions


def heuristic_execution(waste_env: Union[ToxicWasteEnvV1, ToxicWasteEnvV2], agent_models: List[GreedyAgent]) -> List[int]:

	actions = []
	obs = waste_env.create_observation()
	for model in agent_models:
		actions.append(model.act(obs))
	
	return actions

def train_astro_model(waste_env: ToxicWasteEnvV2, astro_model: CentralizedMADQN, agent_models: List[GreedyAgent], waste_order: List,
					  num_iterations: int, max_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float, final_eps: float,
					  eps_type: str, rng_seed: int, logger: logging.Logger, exploration_decay: float = 0.99, warmup: int = 0, target_freq: int = 1000,
					  train_freq: int = 10, summary_frequency: int = 1000, greedy_actions: bool = True, cycle: int = 0,
					  debug_mode: bool = False, interactive: bool = False, anneal_cool: float = 0.9) -> List:
	
	history = []
	random.seed(rng_seed)
	np.random.seed(rng_seed)
	rng_gen = np.random.default_rng(rng_seed)
	n_agents = waste_env.n_players
	n_joint_actions = waste_env.action_space[0].n * waste_env.n_players
	if interactive:
		stop_thread = threading.Event()
		command_thread = threading.Thread(target=input_callback, args=(waste_env, stop_thread))
		command_thread.start()
	
	obs, *_ = waste_env.reset()
	if waste_env.use_render:
		waste_env.render()
	dqn_model = astro_model.madqn
	obs_shape = obs.shape
	if dqn_model.cnn_layer:
		dqn_model.init_network_states(rng_seed, obs.reshape((obs_shape[0], 1, *obs_shape[1:])), optim_learn_rate)
	else:
		dqn_model.init_network_states(rng_seed, obs, optim_learn_rate)
	
	start_time = time.time()
	epoch = 0
	start_record_it = cycle * num_iterations
	start_record_epoch = cycle * max_timesteps
	episode_start = epoch
	episode_rewards = 0
	episode_q_vals = []
	
	for it in range(num_iterations):
		logger.info("Iteration %d out of %d" % (it + 1, num_iterations))
		episode_history = []
		done = False
		while not done:
			# interact with environment
			if eps_type == 'epoch':
				eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, epoch, max_timesteps)
			else:
				eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, it, num_iterations)
			if rng_gen.random() < eps:
				actions = waste_env.action_space.sample()
			else:
				dqn_model = astro_model.madqn
				if dqn_model.cnn_layer:
					q_values = dqn_model.q_network.apply(dqn_model.online_state.params, obs.reshape((obs_shape[0], 1, *obs_shape[1:])))[0]
				else:
					q_values = dqn_model.q_network.apply(dqn_model.online_state.params, obs)
				
				if greedy_actions:
					joint_action = q_values.argmax(axis=-1)
				else:
					pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
					pol = pol / pol.sum()
					joint_action = rng_gen.choice(range(n_joint_actions), p=pol)
				actions = convert_joint_act(joint_action, n_agents, waste_env.action_space[0].n)
				episode_q_vals += [float(q_values[int(joint_action)])]
			
			if debug_mode:
				logger.info('Environment current state')
				logger.info(waste_env.get_env_log())
				logger.info('Player actions: %s' % str([Actions(act).name for act in actions]))
			
			next_obs, rewards, terminated, timeout, infos = waste_env.step(actions)
			if waste_env.use_render:
				waste_env.render()
			if debug_mode:
				logger.info('Player rewards: %s' % str(rewards))
			
			episode_history += [get_history_entry(waste_env.create_observation(), actions, n_agents)]
			step_reward = sum(rewards) / astro_model.num_agents
			episode_rewards += step_reward
			if dqn_model.use_summary:
				astro_model.madqn.summary_writer.add_scalar("charts/reward", step_reward, epoch)
			
			if terminated:
				finished = np.ones(n_agents)
			else:
				finished = np.zeros(n_agents)
				
			# store new samples
			astro_model.replay_buffer.add(obs, next_obs, np.array(actions), rewards, finished, [infos])
			
			# update Q-network and target network
			astro_model.update_dqn_models(batch_size, epoch, start_time, target_freq, tau, summary_frequency, train_freq, warmup)
			
			obs = next_obs
			epoch += 1
			sys.stdout.flush()
			if terminated or timeout:
				episode_len = epoch - episode_start
				if dqn_model.use_summary:
					dqn_model.summary_writer.add_scalar("charts/episode_q_vals", sum(episode_q_vals), epoch)
					dqn_model.summary_writer.add_scalar("charts/mean_episode_q_vals", sum(episode_q_vals) / len(episode_q_vals), epoch + start_record_epoch)
					dqn_model.summary_writer.add_scalar("charts/episode_return", episode_rewards, it + start_record_it)
					dqn_model.summary_writer.add_scalar("charts/mean_episode_return", episode_rewards / episode_len, it + start_record_it)
					dqn_model.summary_writer.add_scalar("charts/episodic_length", episode_len, it + start_record_it)
					dqn_model.summary_writer.add_scalar("charts/epsilon", eps, it + start_record_it)
				obs, *_ = waste_env.reset()
				if waste_env.use_render:
					waste_env.render()
				episode_rewards = 0
				episode_q_vals = []
				episode_start = epoch
				done = True
				history += [episode_history]
				[model.reset(waste_order, dict([(idx, waste_env.objects[idx].position) for idx in range(waste_env.n_objects)])) for model in agent_models]
	
	if interactive:
		stop_thread.set()
	return history


def train_astro_model_v2(waste_env: ToxicWasteEnvV2, astro_model: CentralizedMADQN, agent_models: List[GreedyAgent], waste_order: List,
						 num_iterations: int, max_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float, final_eps: float,
						 eps_type: str, rng_seed: int, logger: logging.Logger, exploration_decay: float = 0.99, warmup: int = 0, target_freq: int = 1000,
						 train_freq: int = 10, summary_frequency: int = 1000, greedy_actions: bool = True, cycle: int = 0,
						 debug_mode: bool = False, interactive: bool = False, anneal_cool: float = 0.9) -> List:
	
	def get_model_obs(raw_obs: Union[np.ndarray, Dict]) -> Tuple[np.ndarray, np.ndarray]:
		conv_obs = []
		arr_obs = []
		
		if waste_env.use_joint_obs:
			if isinstance(raw_obs[0], dict):
				conv_obs = raw_obs[0]['conv']
				arr_obs = np.array(raw_obs[0]['array'])
			else:
				conv_obs = raw_obs[0][0].reshape(1, *raw_obs[0].shape)
				arr_obs = raw_obs[0][1:]
			return conv_obs.reshape(1, *conv_obs.shape), arr_obs
		
		else:
			n_agents = len(raw_obs)
			for idx in range(n_agents):
				if isinstance(raw_obs[idx], dict):
					conv_obs += [raw_obs[idx]['conv']]
					arr_obs += [np.array(raw_obs[idx]['array'])]
				else:
					conv_obs += [raw_obs[idx][0].reshape(1, *raw_obs[idx][0].shape)]
					arr_obs += [raw_obs[idx][1:]]
			conv_obs = np.array(conv_obs)
			return conv_obs.reshape(1, *conv_obs.shape), np.array(arr_obs[0])

	history = []
	if interactive:
		stop_thread = threading.Event()
		command_thread = threading.Thread(target=input_callback, args=(waste_env, stop_thread))
		command_thread.start()
	decision_rng_gen = np.random.default_rng(rng_seed)
	anneal_rng_gen = np.random.default_rng(rng_seed + 1)
	n_agents = waste_env.n_players
	n_joint_actions = waste_env.action_space[0].n * waste_env.n_players
	
	obs, *_ = waste_env.reset()
	dqn_model = astro_model.madqn
	v2_obs = get_model_obs(obs)
	dqn_model.init_network_states(rng_seed, (v2_obs[0], v2_obs[1].reshape((1, 1))), optim_learn_rate)
	
	if waste_env.use_render:
		waste_env.render()
	
	start_time = time.time()
	epoch = 0
	start_record_it = cycle * num_iterations
	start_record_epoch = cycle * max_timesteps
	episode_start = epoch
	episode_rewards = 0
	episode_q_vals = []
	temp = 1.0
	eps = initial_eps
	
	for it in range(num_iterations):
		logger.info("Iteration %d out of %d" % (it + 1, num_iterations))
		logger.info(waste_env.get_env_log())
		episode_history = []
		done = False
		anneal = (anneal_rng_gen.random() < temp)
		while not done:
			# interact with environment
			if anneal:
				actions = heuristic_execution(waste_env, agent_models)
			else:
				if eps_type == 'epoch':
					eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, epoch, max_timesteps)
				else:
					eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, it, num_iterations)
				
				actions = model_execution(dqn_model, eps, get_model_obs, greedy_actions, n_agents, n_joint_actions, obs, decision_rng_gen,
										  waste_env, episode_q_vals)
			
			if debug_mode:
				logger.info('Environment current state')
				logger.info(waste_env.get_env_log())
				logger.info('Player actions: %s' % str([Actions(act).name for act in actions]))
			
			next_obs, rewards, terminated, timeout, infos = waste_env.step(actions)
			if waste_env.use_render:
				waste_env.render()
			if debug_mode:
				logger.info('Player rewards: %s' % str(rewards))
			
			episode_history += [get_history_entry(waste_env.create_observation(), actions, n_agents)]
			step_reward = sum(rewards) / astro_model.num_agents
			episode_rewards += step_reward
			if dqn_model.use_summary:
				astro_model.madqn.summary_writer.add_scalar("charts/reward", step_reward, epoch)
			
			if terminated:
				finished = np.ones(n_agents)
			else:
				finished = np.zeros(n_agents)
			
			# store new samples
			if isinstance(obs[0], dict):
				next_v2_obs = get_model_obs(next_obs)
				astro_model.replay_buffer.add({'conv': v2_obs[0], 'array': v2_obs[1]}, {'conv': next_v2_obs[0], 'array': next_v2_obs[1]},
											  np.array(actions), np.array(step_reward), finished[0], [infos])
			else:
				astro_model.replay_buffer.add(obs, next_obs, np.array(actions), rewards, finished[0], [infos])
				
				# update Q-network and target network
				# astro_model.update_dqn_models(batch_size, epoch, start_time, target_freq, tau, summary_frequency, train_freq, warmup, waste_env.action_space[0].n)
			
			obs = next_obs
			epoch += 1
			if terminated or timeout:
				episode_len = epoch - episode_start
				dqn_model = astro_model.madqn
				if dqn_model.use_summary:
					sum_qs = sum(episode_q_vals)
					n_qs = len(episode_q_vals)
					dqn_model.summary_writer.add_scalar("charts/episode_q_vals", sum_qs, epoch)
					dqn_model.summary_writer.add_scalar("charts/mean_episode_q_vals", sum_qs / n_qs if n_qs > 0 else 0, epoch + start_record_epoch)
					dqn_model.summary_writer.add_scalar("charts/episode_return", episode_rewards, it + start_record_it)
					dqn_model.summary_writer.add_scalar("charts/mean_episode_return", episode_rewards / episode_len, it + start_record_it)
					dqn_model.summary_writer.add_scalar("charts/episodic_length", episode_len, it + start_record_it)
					dqn_model.summary_writer.add_scalar("charts/epsilon", eps, it + start_record_it)
					dqn_model.summary_writer.add_scalar("charts/anneal_temp", temp, it + start_record_it)
					dqn_model.summary_writer.add_scalar("charts/iteration", it, it + start_record_it)
				obs, *_ = waste_env.reset()
				if waste_env.use_render:
					waste_env.render()
				else:
					waste_env.close_render()
				episode_rewards = 0
				episode_q_vals = []
				episode_start = epoch
				done = True
				history += [episode_history]
				[model.reset(waste_order, dict([(idx, waste_env.objects[idx].position) for idx in range(waste_env.n_objects)])) for model in agent_models]
		
		# update Q-network and target network
		astro_model.update_dqn_models(batch_size, it + 1, start_time, target_freq, tau, summary_frequency, train_freq, 1, waste_env.action_space[0].n)
		temp *= anneal_cool
	
	if interactive:
		stop_thread.set()
	return history


def main():
	parser = argparse.ArgumentParser(description='Train DQN model for Astro waste disposal game.')

	# Multi-agent DQN params
	parser.add_argument('--nagents', dest='n_agents', type=int, required=True, help='Number of agents in the environment')
	parser.add_argument('--architecture', dest='architecture', type=str, required=True, help='DQN architecture to use from the architectures yaml')
	parser.add_argument('--buffer', dest='buffer_size', type=int, required=True, help='Size of the replay buffer in the DQN')
	parser.add_argument('--gamma', dest='gamma', type=float, required=False, default=0.99, help='Discount factor for agent\'s future rewards')
	parser.add_argument('--gpu', dest='use_gpu', action='store_true', help='Flag that signals the use of gpu for the training')
	parser.add_argument('--ddqn', dest='use_ddqn', action='store_true', help='Flag that signals the use of a Double DQN')
	parser.add_argument('--vdn', dest='use_vdn', action='store_true', help='Flag that signals the use of a VDN DQN architecture')
	parser.add_argument('--cnn', dest='use_cnn', action='store_true', help='Flag that signals the use of a CNN as entry for the DQN architecture')
	parser.add_argument('--dueling', dest='dueling_dqn', action='store_true', help='Flag that signals the use of a Dueling DQN architecture')
	parser.add_argument('--tensorboard', dest='use_tensorboard', action='store_true',
						help='Flag the signals the use of a tensorboard summary writer. Expects argument --tensorboardDetails to be present')
	parser.add_argument('--tensorboardDetails', dest='tensorboard_details', nargs='+', required=False, default=None,
						help='List with the details for the tensorboard summary writer: <log_dirname: str>, <queue_size :int>, <flush_time: int>, <suffix: str>'
							 ' Use only in combination with --tensorboard option')
	
	# Train parameters
	# parser.add_argument('--cycles', dest='n_cycles', type=int, required=True,
	# 					help='Number of training cycles, each cycle spawns the field with a different food items configurations.')
	parser.add_argument('--iterations', dest='n_iterations', type=int, required=True, help='Number of iterations to run training')
	parser.add_argument('--batch', dest='batch_size', type=int, required=True, help='Number of samples in each training batch')
	parser.add_argument('--train-freq', dest='train_freq', type=int, required=True, help='Number of epochs between each training update')
	parser.add_argument('--target-freq', dest='target_freq', type=int, required=True, help='Number of epochs between updates to target network')
	parser.add_argument('--alpha', dest='learn_rate', type=float, required=False, default=2.5e-4, help='Learn rate for DQN\'s Q network')
	parser.add_argument('--tau', dest='target_learn_rate', type=float, required=False, default=2.5e-6, help='Learn rate for the target network')
	parser.add_argument('--init-eps', dest='initial_eps', type=float, required=False, default=1., help='Exploration rate when training starts')
	parser.add_argument('--final-eps', dest='final_eps', type=float, required=False, default=0.05, help='Minimum exploration rate for training')
	parser.add_argument('--eps-decay', dest='eps_decay', type=float, required=False, default=0.5, help='Decay rate for the exploration update')
	parser.add_argument('--cycle-eps-decay', dest='cycle_eps_decay', type=float, required=False, default=0.95, help='Decay rate for the exploration update')
	parser.add_argument('--eps-type', dest='eps_type', type=str, required=False, default='log', choices=['linear', 'exp', 'log', 'epoch'],
						help='Type of exploration rate update to use: linear, exponential (exp), logarithmic (log), epoch based (epoch)')
	parser.add_argument('--warmup-steps', dest='warmup', type=int, required=False, default=10000, help='Number of epochs to pass before training starts')
	parser.add_argument('--tensorboard-freq', dest='tensorboard_freq', type=int, required=False, default=1,
						help='Number of epochs between each log in tensorboard. Use only in combination with --tensorboard option')
	parser.add_argument('--restart', dest='restart_train', action='store_true',
						help='Flag that signals that train is suppose to restart from a previously saved point.')
	parser.add_argument('--restart-info', dest='restart_info', type=str, nargs='+', required=False, default=None,
						help='List with the info required to recover previously saved model and restart from same point: '
							 '<model_dirname: str> <model_filename: str> <last_cycle: int> Use only in combination with --restart option')
	parser.add_argument('--debug', dest='debug', action='store_true', help='Flag signalling debug mode for model training')
	parser.add_argument('--fraction', dest='fraction', type=str, default='0.5', help='Fraction of JAX memory pre-compilation')
	parser.add_argument('--anneal-temp', dest='anneal_temp', type=float, default=ANNEAL_TEMP, help='Temperature for the heuristic annealing')

	# Environment parameters
	parser.add_argument('--version', dest='env_version', type=int, required=True, help='Environment version to use')
	parser.add_argument('--game-levels', dest='game_levels', type=str, required=True, nargs='+', help='Level to train Astro in.')
	parser.add_argument('--max-env-steps', dest='max_steps', type=int, required=True, help='Maximum number of steps for environment timeout')
	parser.add_argument('--field-size', dest='field_size', type=int, required=True, nargs='+', help='Number of rows and cols in field')
	parser.add_argument('--slip', dest='has_slip', action='store_true', help='')
	parser.add_argument('--require_facing', dest='require_facing', action='store_true', help='')
	parser.add_argument('--agent-centered', dest='centered_obs', action='store_true', help='')
	parser.add_argument('--use-encoding', dest='use_encoding', action='store_true', help='')
	parser.add_argument('--layer-obs', dest='use_layers', action='store_true', help='Environment observation in layer organization')
	parser.add_argument('--render', dest='use_render', action='store_true', help='Flag signaling the use of a render')
	parser.add_argument('--render-mode', dest='render_mode', type=str, nargs='+', required=False, default=None,
						help='List of render modes for the environment')
	
	args = parser.parse_args()
	# DQN args
	n_agents = args.n_agents
	architecture = args.architecture
	buffer_size = args.buffer_size
	gamma = args.gamma
	use_gpu = args.use_gpu
	dueling_dqn = args.dueling_dqn
	use_ddqn = args.use_ddqn
	use_vdn = args.use_vdn
	use_cnn = args.use_cnn
	use_tensorboard = args.use_tensorboard
	# [log_dir: str, queue_size: int, flush_interval: int, filename_suffix: str]
	tensorboard_details = args.tensorboard_details
	
	# Train args
	n_iterations = args.n_iterations
	batch_size = args.batch_size
	train_freq = args.train_freq
	target_freq = args.target_freq
	learn_rate = args.learn_rate
	target_update_rate = args.target_learn_rate
	initial_eps = args.initial_eps
	final_eps = args.final_eps
	eps_decay = args.eps_decay
	eps_type = args.eps_type
	warmup = args.warmup
	tensorboard_freq = args.tensorboard_freq
	debug = args.debug
	temp_anneal = args.anneal_temp
	
	# Astro environment args
	env_version = args.env_version
	game_levels = args.game_levels
	field_size = tuple(args.field_size) if len(args.field_size) == 2 else tuple([args.field_size[0], args.field_size[0]])
	has_slip = args.has_slip
	max_episode_steps = args.max_steps
	facing = args.require_facing
	centered_obs = args.centered_obs
	use_encoding = args.use_encoding
	render_mode = args.render_mode
	use_render = args.use_render
	
	os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = args.fraction
	if not use_gpu:
		jax.default_device(jax.devices("cpu")[0])
	
	now = datetime.now()
	home_dir = Path(__file__).parent.absolute().parent.absolute()
	log_dir = home_dir / 'logs'
	models_dir = home_dir / 'models'
	configs_dir = Path(__file__).parent.absolute() / 'env' / 'data' / 'configs'
	model_path = models_dir / 'astro_disposal_dqn' / now.strftime("%Y%m%d-%H%M%S")
	rng_gen = np.random.default_rng(RNG_SEED)
	
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
			pool_padding = arch_data[architecture]['pool_padding']
			cnn_properties = [n_conv_layers, cnn_size, cnn_kernel, cnn_strides, pool_window, pool_strides, pool_padding]
	
	wandb.init(project='astro-toxic-waste', entity='miguel-faria',
			   config={
				   "agent_type": "joint_policy",
				   "env_version": "v1" if env_version == 1 else "v2",
				   "agents": n_agents,
				   "online_learing_rate": learn_rate,
				   "target_learning_rate": target_update_rate,
				   "discount": gamma,
				   "eps_decay": eps_type,
				   "iterations": n_iterations
			   },
			   name=('joint_policy_' + now.strftime("%Y%m%d-%H%M%S")),
			   sync_tensorboard=True)

	for game_level in game_levels:
		log_filename = ('train_astro_disposal_multi_dqn_%s' % game_level + '_' + now.strftime("%Y%m%d-%H%M%S"))
		logging.basicConfig(filename=(log_dir / (log_filename + '_log.txt')), filemode='w', format='%(name)s %(asctime)s %(levelname)s:\t%(message)s',
							level=logging.INFO)
		logger = logging.getLogger('INFO')
		err_logger = logging.getLogger('ERROR')
		handler = logging.StreamHandler(sys.stderr)
		handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
		err_logger.addHandler(handler)
		Path.mkdir(model_path, parents=True, exist_ok=True)
		with open(configs_dir / 'layouts' / (game_level + '.yaml')) as config_file:
			objects = yaml.safe_load(config_file)['objects']
			n_objects = len(objects['unspecified']) if env_version == 1 else sum([len(objects[key]['ids']) for key in objects.keys() if key != 'unspecified'])
		
		logger.info('#######################################')
		logger.info('Starting Astro Waste Disposal DQN Train')
		logger.info('#######################################')
		logger.info('Level %s setup' % game_level)
		if env_version == 1:
			env = ToxicWasteEnvV1(field_size, game_levels[0], n_agents, n_objects, max_episode_steps, RNG_SEED, facing, args.use_layers, centered_obs,
								  use_encoding, render_mode, slip=has_slip, use_render=use_render, joint_obs=True)
		else:
			env = ToxicWasteEnvV2(field_size, game_levels[0], n_agents, n_objects, max_episode_steps, RNG_SEED, facing, centered_obs, render_mode,
								  slip=has_slip, is_train=True, use_render=use_render, joint_obs=True)
		
		obs, *_ = env.reset(seed=RNG_SEED)
		
		logger.info('Getting human behaviour model')
		agent_models = []
		for player in env.players:
			if env_version == 1:
				agent_models.append(GreedyAgent(player.position, player.orientation, player.name,
												dict([(idx, env.objects[idx].position) for idx in range(n_objects)]), RNG_SEED, env.field, env_version,
												agent_type=player.agent_type))
			else:
				agent_models.append(GreedyAgent(player.position, player.orientation, player.name,
												dict([(idx, env.objects[idx].position) for idx in range(n_objects)]), RNG_SEED, env.field, env_version,
												env.door_pos, agent_type=player.agent_type))
		
		logger.info('Train setup')
		waste_idx = []
		for obj in env.objects:
			waste_idx.append(env.objects.index(obj))
		waste_seqs = list(permutations(waste_idx))
		waste_order = list(rng_gen.choice(np.array(waste_seqs)))
		for model in agent_models:
			model.waste_order = waste_order
		
		logger.info('Creating DQN and starting train')
		tensorboard_details[0] = tensorboard_details[0] + '/astro_disposal_' + game_level + '_' + now.strftime("%Y%m%d-%H%M%S")
		tensorboard_details += ['astro_' + game_level]
		
		astro_dqn = CentralizedMADQN(n_agents if not env.use_joint_obs else 1, env.action_space[0].n, n_layers, convert_joint_act, nn.relu, layer_sizes,
									 buffer_size, gamma, env.action_space, env.observation_space, use_gpu, dueling_dqn, use_ddqn, use_cnn, (env_version == 2),
									 False, use_tensorboard=use_tensorboard, tensorboard_data=tensorboard_details, cnn_properties=cnn_properties)
		if env_version == 1:
			train_astro_model(env, astro_dqn, agent_models, waste_order, n_iterations, max_episode_steps * n_iterations, batch_size, learn_rate,
							  target_update_rate, initial_eps, final_eps, eps_type, RNG_SEED, logger, eps_decay, warmup, target_freq, train_freq,
							  tensorboard_freq, debug_mode=debug, interactive=INTERACTIVE_SESSION)
		else:
			train_astro_model_v2(env, astro_dqn, agent_models, waste_order, n_iterations, max_episode_steps * n_iterations, batch_size, learn_rate,
								 target_update_rate, initial_eps, final_eps, eps_type, RNG_SEED, logger, eps_decay, warmup, target_freq, train_freq,
								 tensorboard_freq, debug_mode=debug, interactive=INTERACTIVE_SESSION, anneal_cool=ANNEAL_TEMP)

		logger.info('Saving model and history list')
		Path.mkdir(model_path, parents=True, exist_ok=True)
		astro_dqn.save_model(game_level, model_path, logger)
		# obs_path = model_path / (game_level + '.json')
		# with open(obs_path, "w") as of:
		# 	of.write(json.dumps(history))


if __name__ == '__main__':
	main()
