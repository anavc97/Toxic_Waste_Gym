#! /usr/bin/env python
import sys
import time
import gymnasium
import jax
import numpy as np
import logging

from pathlib import Path
from src.algos.dqn import DQNetwork, EPS_TYPE
from src.utilities.buffers import ReplayBuffer
from typing import List, Callable, Optional
from datetime import datetime
from gymnasium.spaces import Space
from wandb.wandb_run import Run


class SingleAgentDQN(object):
	
	_agent_dqn: DQNetwork
	_replay_buffer: ReplayBuffer
	_perform_tracker: Run
	_use_tracker: bool
	_use_v2: bool

	def __init__(self, action_dim: int, num_layers: int, act_function: Callable, layer_sizes: List[int], buffer_size: int, gamma: float, action_space: Space,
	             observation_space: Space, use_gpu: bool, dueling_dqn: bool = False, use_ddqn: bool = False, use_cnn: bool = False,
	             handle_timeout: bool = False, use_tensorboard: bool = False, tracker: Optional[Run] = None, cnn_properties: List[int] = None,
	             use_v2: bool = False):
		
		self._use_tracker = use_tensorboard
		if use_tensorboard:
			self._perform_tracker = tracker
		self._use_v2 = use_v2
		self._agent_dqn = DQNetwork(action_dim, num_layers, act_function, layer_sizes, gamma, dueling_dqn, use_ddqn, use_cnn, use_tensorboard, cnn_properties)
		self._replay_buffer = ReplayBuffer(buffer_size, observation_space, action_space, "cuda" if use_gpu else "cpu",
										   handle_timeout_termination=handle_timeout)
	
	########################
	### Class Properties ###
	########################
	@property
	def agent_dqn(self) -> DQNetwork:
		return self._agent_dqn
	
	@property
	def use_tracker(self) -> bool:
		return self._use_tracker
	
	@property
	def replay_buffer(self) -> ReplayBuffer:
		return self._replay_buffer
	
	#####################
	### Class Methods ###
	#####################
	def train(self, env: gymnasium.Env, num_iterations: int, max_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float,
			  final_eps: float, eps_type: str, rng_seed: int, exploration_decay: float = 0.99, warmup: int = 0, target_freq: int = 1000, train_freq: int = 10,
			  summary_frequency: int = 1):
		
		np.random.seed(rng_seed)
		rng_gen = np.random.default_rng(rng_seed)
		
		obs, *_ = env.reset()
		self._agent_dqn.init_network_states(rng_seed, obs, optim_learn_rate)
		
		start_time = time.time()
		epoch = 0
		history = []
		
		for it in range(num_iterations):
			done = False
			episode_rewards = 0
			episode_start = epoch
			episode_history = []
			print("Iteration %d out of %d" % (it + 1, num_iterations))
			while not done:
				print("Epoch %d" % (epoch + 1))
				
				# interact with environment
				if eps_type == 'epoch':
					eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, epoch, max_timesteps)
				else:
					eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, it, num_iterations)
				
				if rng_gen.random() < eps:
					action = np.array([env.action_space.sample()])
				else:
					if self._agent_dqn.cnn_layer:
						q_values = self._agent_dqn.q_network.apply(self._agent_dqn.online_state.params, obs.reshape((1, *obs.shape)))[0]
					else:
						q_values = self._agent_dqn.q_network.apply(self._agent_dqn.online_state.params, obs)
					action = q_values.argmax(axis=-1)
					action = jax.device_get(action)
				next_obs, reward, finished, timeout, info = env.step(action)
				episode_rewards += reward
				episode_history += [obs, action]
				
				# store new samples
				self._replay_buffer.add(obs, next_obs, action, np.array([reward]), np.array(finished), [info])
				obs = next_obs
				
				# update Q-network and target network
				self.update_dqn_models(batch_size, epoch, start_time, target_freq, tau, summary_frequency, train_freq, warmup)
				
				epoch += 1
				sys.stdout.flush()
				if finished:
					obs, _ = env.reset()
					done = True
					history += [episode_history]
					if self._use_tracker:
						self._perform_tracker.log({
								"charts/episodic_return": episode_rewards,
								"charts/episodic_length": epoch - episode_start,
								"charts/epsilon": eps
						}, epoch)
						print("Episode over:\tReward: %f\tLength: %d" % (episode_rewards, epoch - episode_start))
		
		return history
	
	def train_cnn(self, env: gymnasium.Env, num_iterations: int, max_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float,
				  final_eps: float, eps_type: str, rng_seed: int, logger: logging.Logger, exploration_decay: float = 0.99, warmup: int = 0,
				  target_freq: int = 1000, train_freq: int = 10, summary_frequency: int = 1):
		
		np.random.seed(rng_seed)
		rng_gen = np.random.default_rng(rng_seed)

		env.reset()
		obs = env.render()
		self._agent_dqn.init_network_states(rng_seed, obs, optim_learn_rate)
		
		start_time = time.time()
		epoch = 0
		history = []
		
		for it in range(num_iterations):
			done = False
			episode_rewards = 0
			episode_start = epoch
			episode_history = []
			logger.info("Iteration %d out of %d" % (it + 1, num_iterations))
			while not done:
				logger.debug("Epoch %d" % (epoch + 1))
				
				# interact with environment
				if eps_type == 'epoch':
					eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, epoch, max_timesteps)
				else:
					eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, it, num_iterations)
				
				if rng_gen.random() < eps:
					action = np.array(env.action_space.sample())
				else:
					q_values = self._agent_dqn.q_network.apply(self._agent_dqn.online_state.params, obs)
					action = q_values.argmax(axis=-1)
					action = jax.device_get(action)
				_, reward, finished, timeout, info, *_ = env.step(action)
				next_obs = env.render()
				episode_rewards += reward
				episode_history += [obs, action]
				
				# store new samples
				self._replay_buffer.add(obs, next_obs, action, reward, finished, info)
				obs = next_obs
				
				# update Q-network and target network
				self.update_dqn_model(batch_size, epoch, start_time, summary_frequency, target_freq, tau, train_freq, warmup)
				
				epoch += 1
				sys.stdout.flush()
				if finished:
					env.reset()
					obs = env.render()
					done = True
					history += [episode_history]
					if self._use_tracker:
						self._perform_tracker.log({
								"charts/episodic_return": episode_rewards,
								"charts/episodic_length": epoch - episode_start,
								"charts/epsilon":         eps
						}, epoch)
						logger.debug("Episode over:\tReward: %f\tLength: %d" % (episode_rewards, epoch - episode_start))
		
		return history
	
	def update_dqn_models(self, batch_size: int, epoch: int, start_time: float, target_freq: int, tau: float, tensorboard_frequency: int,
						  train_freq: int, warmup: int):
		
		if epoch >= warmup:
			if epoch % train_freq == 0:
				data = self._replay_buffer.sample(batch_size)
				
				if self._use_v2:
					if isinstance(data.observations, dict):
						obs_conv = data.observations['conv']
						obs_array = data.observations['array']
						next_obs_conv = data.next_observations['conv']
						next_obs_array = data.next_observations['array']
					else:
						obs_conv = data.observations[0]
						obs_array = data.observations[1]
						next_obs_conv = data.next_observations[0]
						next_obs_array = data.next_observations[1]
					actions = data.actions
					rewards = data.rewards
					dones = data.dones
					self.agent_dqn.update_online_model((obs_conv, obs_array[:, 0]), actions, (next_obs_conv, next_obs_array[:, 0]),
													   rewards, dones, epoch, start_time, tensorboard_frequency)
				
				else:
					observations = data.observations
					next_observations = data.next_observations
					actions = data.actions
					rewards = data.rewards
					dones = data.dones
					self.agent_dqn.update_online_model(observations, actions, next_observations, rewards, dones, epoch, start_time, tensorboard_frequency)
			
			if epoch % target_freq == 0:
				self.agent_dqn.update_target_model(tau)
	
	def save_model(self, filename: str, model_dir: Path, logger: logging.Logger) -> None:
		self._agent_dqn.save_model(filename, model_dir, logger)
	
	def load_model(self, filename: str, model_dir: Path, logger: logging.Logger, obs_shape: tuple) -> None:
		self._agent_dqn.load_model(filename, model_dir, logger, obs_shape)