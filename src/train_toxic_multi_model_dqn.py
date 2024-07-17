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
import json
import traceback

from algos.dqn import EPS_TYPE, DQNetwork
from algos.multi_model_madqn import MultiAgentDQN
from env.toxic_waste_env_v1 import ToxicWasteEnvV1
from env.toxic_waste_env_v2 import ToxicWasteEnvV2, Actions
from env.astro_greedy_agent import GreedyAgent
from pathlib import Path
from typing import List, Union, Dict
from datetime import datetime
from itertools import permutations


RNG_SEED = 21062023
ROBOT_NAME = 'astro'
INTERACTIVE_SESSION = False
ANNEAL_DECAY = 0.95
RESTART_WARMUP = 5
MOVE_PENALTY = -1


def get_history_entry(obs: ToxicWasteEnvV2.Observation, actions: List[int], n_agents: int) -> List:
	
	entry = []
	for a_idx in range(n_agents):
		agent = obs.players[a_idx]
		action = actions[a_idx]
		entry += [agent.id, agent.position, agent.orientation, agent.held_objects, action]
	
	return entry


def input_callback(env: Union[ToxicWasteEnvV1, ToxicWasteEnvV2], stop_flag: bool):
	
	try:
		while not stop_flag:
			command = input('Interactive commands:\n\trender - display renderization of the interaction\n\tstop_render - stops the renderization\nCommand: ')
			if command == 'render':
				env.use_render = True
			elif command == 'stop_render':
				if env.use_render:
					env.close_render()
					env.use_render = False
	
	except KeyboardInterrupt as ki:
		return


def get_model_obs(raw_obs: Union[np.ndarray, Dict]) -> np.ndarray:
	if isinstance(raw_obs, dict):
		model_obs = np.array([raw_obs['conv'].reshape(1, *raw_obs['conv'].shape), np.array(raw_obs['array'])],
							 dtype=object)
	else:
		model_obs = np.array([raw_obs[0].reshape(1, *raw_obs[0].shape), raw_obs[1:]], dtype=object)
	
	return model_obs


def model_execution(agent_ids: List[str], ma_model: MultiAgentDQN, eps: float, greedy_actions: bool, n_agents: int, obs: np.ndarray, rng_gen: np.random.Generator,
					waste_env: Union[ToxicWasteEnvV1, ToxicWasteEnvV2], episode_q_vals: List[List[float]]) -> List[int]:

	if rng_gen.random() < eps:
		actions = waste_env.action_space.sample()
	else:
		actions = []
		for a_idx in range(n_agents):
			a_id = agent_ids[a_idx]
			dqn_model = ma_model.agent_dqns[a_id]
			model_obs = get_model_obs(obs[a_idx])
			q_values = dqn_model.q_network.apply(dqn_model.online_state.params, model_obs[0], model_obs[1])[0]
			
			if greedy_actions:
				action = q_values.argmax(axis=-1)
			else:
				pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
				pol = pol / pol.sum()
				action = rng_gen.choice(range(waste_env.action_space[0].n), p=pol)
			episode_q_vals[a_idx] += float(q_values[int(action)])
			actions += [int(jax.device_get(action))]
	
	return actions


def heuristic_execution(waste_env: Union[ToxicWasteEnvV1, ToxicWasteEnvV2], agent_models: List[GreedyAgent], train_only_movement: bool = False) -> List[int]:

	actions = []
	obs = waste_env.create_observation()
	for model in agent_models:
		actions.append(model.act(obs, train_only_movement))
	
	return actions


def train_astro_model(agents_ids: List[str], waste_env: ToxicWasteEnvV2, astro_model: MultiAgentDQN, human_model: List[GreedyAgent], waste_order: List,
					  num_iterations: int, max_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float, final_eps: float,
					  eps_type: str, rng_seed: int, logger: logging.Logger, exploration_decay: float = 0.99, warmup: int = 0, target_freq: int = 1000,
					  train_freq: int = 10, summary_frequency: int = 1000, greedy_actions: bool = True, cycle: int = 0,
					  debug_mode: bool = False, render: bool = False) -> List:
	
	history = []
	random.seed(rng_seed)
	np.random.seed(rng_seed)
	rng_gen = np.random.default_rng(rng_seed)
	n_agents = len(agents_ids)
	
	obs, *_ = waste_env.reset()
	for a_idx in range(n_agents):
		a_id = agents_ids[a_idx]
		dqn_model = astro_model.agent_dqns[a_id]
		if dqn_model.cnn_layer:
			dqn_model.init_network_states(rng_seed, obs[a_idx].reshape(1, *obs[a_idx].shape), optim_learn_rate)
		else:
			dqn_model.init_network_states(rng_seed, obs[a_idx], optim_learn_rate)
	
	start_time = time.time()
	epoch = 0
	start_record_it = cycle * num_iterations
	start_record_epoch = cycle * max_timesteps
	episode_start = epoch
	episode_rewards = [0] * n_agents
	episode_q_vals = [0] * n_agents
	
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
				actions = []
				for a_idx in range(n_agents):
					a_id = agents_ids[a_idx]
					dqn_model = astro_model.agent_dqns[a_id]
					if dqn_model.cnn_layer:
						q_values = dqn_model.q_network.apply(dqn_model.online_state.params, obs[a_idx].reshape((1, *obs[a_idx].shape)))[0]
					else:
						q_values = dqn_model.q_network.apply(dqn_model.online_state.params, obs[a_idx])
					
					if greedy_actions:
						action = q_values.argmax(axis=-1)
					else:
						pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
						pol = pol / pol.sum()
						action = rng_gen.choice(range(waste_env.action_space[a_idx].n), p=pol)
					
					actions += [int(jax.device_get(action))]
					if dqn_model.use_summary:
						dqn_model.summary_writer.add_scalar("charts/episodic_q_vals", float(q_values[int(action)]), epoch)
			if debug_mode:
				logger.info('Environment current state')
				logger.info(waste_env.get_full_env_log())
				logger.info(str(human_model))
				logger.info('Player actions: %s' % str([Actions(act).name for act in actions]))
			next_obs, rewards, terminated, timeout, infos = waste_env.step(actions)
			if debug_mode:
				logger.info('Player rewards: %s' % str(rewards))
			episode_history += [get_history_entry(waste_env.create_observation(), actions, len(agents_ids))]
			for a_idx in range(n_agents):
				episode_rewards[a_idx] += rewards[a_idx]
				dqn_model = astro_model.agent_dqns[agents_ids[a_idx]]
				if dqn_model.use_summary:
					dqn_model.summary_writer.add_scalar("charts/reward", rewards[a_idx], epoch)
			
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
				for a_idx in range(n_agents):
					dqn_model = astro_model.agent_dqns[agents_ids[a_idx]]
					if dqn_model.use_summary:
						dqn_model.summary_writer.add_scalar("charts/mean_episode_q_vals", episode_q_vals[a_idx] / episode_len, epoch + start_record_epoch)
						dqn_model.summary_writer.add_scalar("charts/episode_return", episode_rewards[a_idx], it + start_record_it)
						dqn_model.summary_writer.add_scalar("charts/mean_episode_return", episode_rewards[a_idx] / episode_len, it + start_record_it)
						dqn_model.summary_writer.add_scalar("charts/episodic_length", episode_len, it + start_record_it)
						dqn_model.summary_writer.add_scalar("charts/epsilon", eps, it + start_record_it)
				obs, *_ = waste_env.reset()
				episode_rewards = [0] * n_agents
				episode_q_vals = [0] * n_agents
				episode_start = epoch
				done = True
				history += [episode_history]
				human_model.reset(waste_order, dict([(idx, waste_env.objects[idx].position) for idx in range(waste_env.n_objects)]))
		
	return history


def train_astro_model_v2(waste_env: ToxicWasteEnvV2, multi_agt_model: MultiAgentDQN, heuristic_models: List[GreedyAgent], waste_order: List, num_iterations: int, max_timesteps: int,
						 batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float, final_eps: float, eps_type: str, rng_seed: int, logger: logging.Logger,
						 model_path: Path, game_level: str, chkpt_file: str, chkt_data: dict, exploration_decay: float = 0.99, warmup: int = 0, start_it: int = 0,
						 start_temp: float = 1.0, checkpoint_freq: int = 10, target_freq: int = 1000, train_freq: int = 10, summary_frequency: int = 1000,
						 greedy_actions: bool = True, cycle: int = 0, debug_mode: bool = False, interactive: bool = False, anneal_cool: float = 0.9, restart: bool = False,
						 only_move: bool = True, curriculum_model: List[Union[str, Path]] = None) -> List:

	history = []
	decision_rng_gen = np.random.default_rng(rng_seed)
	anneal_rng_gen = np.random.default_rng(rng_seed + 1)
	n_agents = waste_env.n_players
	agents_ids = [p.name for p in waste_env.players]
	if interactive:
		stop_thread = threading.Event()
		command_thread = threading.Thread(target=input_callback, args=(waste_env, stop_thread))
		command_thread.start()
	
	obs, *_ = waste_env.reset()
	if curriculum_model is not None:
		assert len(curriculum_model) == n_agents
		for a_idx in range(n_agents):
			a_id = agents_ids[a_idx]
			dqn_model = multi_agt_model.agent_dqns[a_id]
			model_obs = get_model_obs(obs[a_idx])
			dqn_model.init_network_states(rng_seed, (model_obs[0], model_obs[1]), optim_learn_rate, curriculum_model[a_idx])
	else:
		for a_idx in range(n_agents):
			a_id = agents_ids[a_idx]
			dqn_model = multi_agt_model.agent_dqns[a_id]
			model_obs = get_model_obs(obs[a_idx])
			dqn_model.init_network_states(rng_seed, (model_obs[0], model_obs[1]), optim_learn_rate)
	if restart:
		warm_anneal_count = RESTART_WARMUP
	
	if waste_env.use_render:
		waste_env.render()
	
	start_time = time.time()
	epoch = 0
	start_record_it = cycle * num_iterations
	start_record_epoch = cycle * max_timesteps
	episode_start = epoch
	warmup_anneal = restart
	temp = start_temp
	eps = initial_eps
	avg_episode_len = []
	avg_episode_q_vals = [[]] * n_agents
	avg_episode_reward = [[]] * n_agents

	for it in range(start_it, num_iterations):
		episode_rewards = [0.0] * n_agents
		episode_q_vals = [0.0] * n_agents
		logger.info("Iteration %d out of %d" % (it + 1, num_iterations))
		logger.info(waste_env.get_full_env_log())
		episode_history = []
		done = False
		anneal = (anneal_rng_gen.random() < temp or warmup_anneal)
		while not done:
			# interact with environment
			if anneal:
				actions = heuristic_execution(waste_env, heuristic_models, only_move)
			else:
				if eps_type == 'epoch':
					eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, epoch, max_timesteps)
				else:
					eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, it, num_iterations)
				
				actions = model_execution(agents_ids, multi_agt_model, eps, greedy_actions, n_agents, obs, decision_rng_gen, waste_env, episode_q_vals)
				
			next_obs, rewards, terminated, timeout, infos = waste_env.step(actions)

			if only_move:
				rewards = np.zeros(waste_env.n_players) if terminated else MOVE_PENALTY * np.ones(waste_env.n_players)
				
			if debug_mode:
				logger.info('Environment current state')
				logger.info(waste_env.get_env_log())
				logger.info('Player actions: %s' % str([Actions(act).name for act in actions]))
				logger.info('Player rewards: %s' % str(rewards))
			
			if waste_env.use_render:
				waste_env.render()
				
			for a_idx in range(n_agents):
				episode_rewards[a_idx] += rewards[a_idx]
				dqn_model = multi_agt_model.agent_dqns[agents_ids[a_idx]]
				if dqn_model.use_summary:
					dqn_model.summary_writer.add_scalar("%s-charts/performance/reward" % agents_ids[a_idx], rewards[a_idx], epoch)
			
			if terminated:
				finished = np.ones(n_agents)
			else:
				finished = np.zeros(n_agents)
			
			# store new samples
			for a_idx in range(n_agents):
				a_id = agents_ids[a_idx]
				multi_agt_model.replay_buffer(a_id).add(obs[a_idx], next_obs[a_idx], np.array(actions[a_idx]), rewards[a_idx], finished[a_idx], [infos])
			
			# # update Q-network and target network
			multi_agt_model.update_dqn_models(batch_size, epoch, start_time, target_freq, tau, summary_frequency, train_freq, warmup)
			
			obs = next_obs
			epoch += 1
			if terminated or timeout:
				episode_len = epoch - episode_start
				avg_episode_len.append(episode_len)
				for a_idx in range(n_agents):
					dqn_model = multi_agt_model.agent_dqns[agents_ids[a_idx]]
					avg_episode_reward[a_idx].append(episode_rewards[a_idx])
					avg_episode_q_vals[a_idx].append(episode_q_vals[a_idx])
					if dqn_model.use_summary:
						dqn_model.summary_writer.add_scalar("%s-charts/performance/episode_q_vals" % agents_ids[a_idx], episode_q_vals[a_idx], it + start_record_it)
						dqn_model.summary_writer.add_scalar("%s-charts/performance/mean_episode_q_vals" % agents_ids[a_idx], np.mean(avg_episode_q_vals[a_idx]), it + start_record_it)
						dqn_model.summary_writer.add_scalar("%s-charts/performance/episode_return" % agents_ids[a_idx], episode_rewards[a_idx], it + start_record_it)
						dqn_model.summary_writer.add_scalar("%s-charts/performance/avg_episode_return" % agents_ids[a_idx], np.mean(avg_episode_reward[a_idx]), it + start_record_it)
						dqn_model.summary_writer.add_scalar("%s-charts/performance/episodic_length" % agents_ids[a_idx], episode_len, it + start_record_it)
						dqn_model.summary_writer.add_scalar("%s-charts/performance/avg_episode_len" % agents_ids[a_idx], np.mean(avg_episode_len), it + start_record_it)
				dqn_model.summary_writer.add_scalar("charts/control/epsilon", eps, it + start_record_it)
				dqn_model.summary_writer.add_scalar("charts/control/anneal_temp", temp, it + start_record_it)
				dqn_model.summary_writer.add_scalar("charts/control/iteration", it, it + start_record_it)
				obs, *_ = waste_env.reset()
				episode_rewards = [0] * n_agents
				episode_q_vals = [0] * n_agents
				episode_start = epoch
				done = True
				history += [episode_history]
				[model.reset(waste_order, dict([(idx, waste_env.objects[idx].position) for idx in range(waste_env.n_objects)])) for model in heuristic_models]
				if warmup_anneal:
					warm_anneal_count -= 1
					warmup_anneal = warm_anneal_count > 0
		
		temp *= anneal_cool
		# update Q-network and target network
		# multi_agt_model.update_dqn_models(batch_size, epoch, start_time, target_freq, tau, summary_frequency, train_freq, warmup)
		
		if it % checkpoint_freq == 0:
			for a_idx in range(n_agents):
				dqn_model = multi_agt_model.agent_dqns[agents_ids[a_idx]]
				dqn_model.save_model('%s-v2_l-%s-checkpoint' % (agents_ids[a_idx], game_level), model_path, logger)
			with open(chkpt_file, 'w') as j_file:
				chkt_data[game_level] = {'iteration': it, 'temp': temp}
				json.dump(chkt_data, j_file)
	
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
	parser.add_argument('--checkpoint-freq', dest='checkpoint_freq', type=int, required=False, default=10,
						help='Number of epochs between each model train checkpointing.')
	parser.add_argument('--restart', dest='restart_train', action='store_true',
						help='Flag that signals that train is suppose to restart from a previously saved point.')
	parser.add_argument('--debug', dest='debug', action='store_true', help='Flag signalling debug mode for model training')
	parser.add_argument('--fraction', dest='fraction', type=str, default='0.5', help='Fraction of JAX memory pre-compilation')
	parser.add_argument('--anneal-decay', dest='anneal_decay', type=float, default=ANNEAL_DECAY, help='Decay value for the heuristic annealing')
	parser.add_argument('--initial-temp', dest='init_temp', type=float, default=1.0, help='Initial value for the annealing temperature.')
	parser.add_argument('--models-dir', dest='models_dir', type=str, default='', help='Directory to store trained models, if left blank stored in default location')
	parser.add_argument('--logs-dir', dest='logs_dir', type=str, default='', help='Directory to store logs, if left blank stored in default location')
	parser.add_argument('--checkpoint-file', dest='checkpoint_file', type=str, required=False, default='', help='File with data from previous training checkpoint')
	parser.add_argument('--buffer-smart-add', dest='buffer_smart_add', action='store_true',
						help='Flag denoting the use of smart sample add to experience replay buffer instead of first-in first-out')
	parser.add_argument('--buffer-method', dest='buffer_method', type=str, required=False, default='uniform', choices=['uniform', 'weighted'],
						help='Method of deciding how to add new experience samples when replay buffer is full')
	parser.add_argument('--use-curriculum', dest='use_curriculum', action='store_true',
						help='Flag that signals training using previously trained models as a starting model')
	parser.add_argument('--curriculum-model', dest='curriculum_model', type=str, nargs='+', default='', help='Path to model to use as a starting model to improve.')
	parser.add_argument('--train-only-movement', dest='only_movement', action='store_true', help='Flag denoting train only of moving in environment')
	parser.add_argument('--has-pick-all', dest='has_pick_all', action='store_true', help='Flag denoting all green and yellow balls have to be picked before human exiting')

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
	checkpoint_freq = args.checkpoint_freq
	debug = args.debug
	decay_anneal = args.anneal_decay
	anneal_temp = args.init_temp
	chkpt_file = args.checkpoint_file
	use_curriculum = args.use_curriculum
	curriculum_models = args.curriculum_model
	only_movement = args.only_movement
	
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
	log_dir = Path(args.logs_dir) if args.logs_dir != '' else home_dir / 'logs'
	models_dir = Path(args.models_dir) / 'models' if args.models_dir != '' else home_dir / 'models'
	configs_dir = Path(__file__).parent.absolute() / 'env' / 'data' / 'configs'
	model_path = models_dir / 'astro_disposal_dqn' / now.strftime("%Y%m%d-%H%M%S")
	rng_gen = np.random.default_rng(RNG_SEED)
	
	if chkpt_file != '' and args.restart_train:
		with open(chkpt_file, 'r') as j_file:
			chkpt_data = json.load(j_file)
	else:
		chkpt_data = {}
		for level in game_levels:
			chkpt_data[level] = {'iteration': 0, 'temp': anneal_temp}
		chkpt_file = str(models_dir / ('v%d_train_checkpoint_data.json' % env_version))
	
	if use_curriculum:
		try:
			assert curriculum_models is not None
			assert all([curriculum_model != '' for curriculum_model in curriculum_models])
		except AssertionError:
			print('Attempt at using curriculum learning but doesn\'t supply a model to use as a starting point')
			return
	else:
		curriculum_models = None
	
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
	
	wandb.init(project='astro-toxic-waste', entity='miguel-faria',
			   config={
					   "agent_type": "independent%s_agents" % ("_vdn" if use_vdn else ""),
					   "env_version": "v1" if env_version == 1 else "v2",
					   "agents": n_agents,
					   "online_learing_rate": learn_rate,
					   "target_learning_rate": target_update_rate,
					   "discount": gamma,
					   "eps_decay_type": eps_type,
					   "eps_decay": eps_decay,
					   "iterations": n_iterations,
					   "buffer_size": buffer_size,
					   "buffer_add": "smart" if args.buffer_smart_add else "plain",
					   "buffer_add_method": args.buffer_method if args.buffer_smart_add else "fifo",
					   "batch_size": batch_size,
					   "online_frequency": train_freq,
					   "target_frequency": target_freq,
					   "architecture": architecture
			   },
			   name=('multi_model%s_' % ("_vdn" if use_vdn else "") + now.strftime("%Y%m%d-%H%M%S")),
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
		try:
			with open(configs_dir / 'layouts' / (game_level + '.yaml')) as config_file:
				objects = yaml.safe_load(config_file)['objects']
				n_objects = len(objects['unspecified']) if env_version == 1 else sum([len(objects[key]['ids']) for key in objects.keys() if key != 'unspecified'])
			
			logger.info('#######################################')
			logger.info('Starting Astro Waste Disposal DQN Train')
			logger.info('#######################################')
			logger.info('Level %s setup' % game_level)
			if env_version == 1:
				env = ToxicWasteEnvV1(field_size, game_level, n_agents, n_objects, max_episode_steps, RNG_SEED, facing,
									  args.use_layers, centered_obs, use_encoding, args.render_mode, slip=has_slip, use_render=use_render)
			else:
				env = ToxicWasteEnvV2(field_size, game_level, n_agents, n_objects, max_episode_steps, RNG_SEED, facing,
									  centered_obs, args.render_mode, slip=has_slip, is_train=True, use_render=use_render, pick_all=args.has_pick_all)
			
			obs, *_ = env.reset(seed=RNG_SEED)
			agents_id = [agent.name for agent in env.players]
			
			heuristic_agents = []
			for player in env.players:
				if env_version == 1:
					heuristic_agents.append(GreedyAgent(player.position, player.orientation, player.name,
													dict([(idx, env.objects[idx].position) for idx in range(n_objects)]), RNG_SEED, env.field, env_version,
													agent_type=player.agent_type))
				else:
					heuristic_agents.append(GreedyAgent(player.position, player.orientation, player.name,
													dict([(idx, env.objects[idx].position) for idx in range(n_objects)]), RNG_SEED, env.field, env_version,
													env.door_pos, agent_type=player.agent_type))
			
			logger.info('Train setup')
			waste_idx = []
			for obj in env.objects:
				waste_idx.append(env.objects.index(obj))
			waste_seqs = list(permutations(waste_idx))
			waste_order = list(rng_gen.choice(np.array(waste_seqs)))
			for model in heuristic_agents:
				model.waste_order = waste_order
			
			logger.info('Creating DQN and starting train')
			tensorboard_details[0] = tensorboard_details[0] + '/astro_disposal_' + game_level + '_' + now.strftime("%Y%m%d-%H%M%S")
			tensorboard_details += ['astro_' + game_level]
			start_it = chkpt_data[game_level]['iteration']
			start_temp = chkpt_data[game_level]['temp']
			
			if use_vdn:
				astro_dqn = MultiAgentDQN(n_agents, agents_id, env.action_space[0].n, n_layers, nn.relu, layer_sizes, buffer_size, gamma, env.action_space,
										  env.observation_space, use_gpu, dueling_dqn, use_ddqn, use_vdn, use_cnn, False,
										  use_tensorboard=use_tensorboard, tensorboard_data=tensorboard_details, use_v2=(env_version == 2),
										  cnn_properties=cnn_properties)
			else:
				astro_dqn = MultiAgentDQN(n_agents, agents_id, env.action_space[0].n, n_layers, nn.relu, layer_sizes, buffer_size, gamma, env.action_space[0],
										  env.observation_space, use_gpu, dueling_dqn, use_ddqn, use_vdn, use_cnn, False,
										  use_tensorboard=use_tensorboard, tensorboard_data=tensorboard_details, use_v2=(env_version == 2),
										  cnn_properties=cnn_properties)
			if env_version == 1:
				train_astro_model(agents_id, env, astro_dqn, heuristic_agents, waste_order, n_iterations, max_episode_steps * n_iterations, batch_size,
								  learn_rate, target_update_rate, initial_eps, final_eps, eps_type, RNG_SEED, logger, eps_decay, warmup, target_freq,
								  train_freq, tensorboard_freq, debug_mode=debug, render=use_render)
			else:
				train_astro_model_v2(env, astro_dqn, heuristic_agents, waste_order, n_iterations, max_episode_steps * n_iterations, batch_size,
									 learn_rate, target_update_rate, initial_eps, final_eps, eps_type, RNG_SEED, logger, models_dir / 'checkpoints', game_level, chkpt_file, chkpt_data,
									 eps_decay, warmup, start_it, start_temp, checkpoint_freq, target_freq, train_freq, tensorboard_freq, debug_mode=debug, greedy_actions=False,
									 interactive=INTERACTIVE_SESSION, anneal_cool=decay_anneal, restart=args.restart_train, curriculum_model=curriculum_models, only_move=only_movement)
	
			logger.info('Saving model and history list')
			Path.mkdir(model_path, parents=True, exist_ok=True)
			astro_dqn.save_models(game_level, model_path, logger)
		
		except KeyboardInterrupt as ks:
			logger.info('Caught keyboard interrupt, cleaning up and closing.')
			wandb.finish()
		
		except Exception as e:
			logger.error("Caught an unexpected exception while training level %s: %s\n%s" % (game_level, str(e), traceback.format_exc()))
	
	wandb.finish()


if __name__ == '__main__':
	main()
