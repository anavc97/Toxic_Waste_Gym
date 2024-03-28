#! /usr/bin/env python
import sys
import time
import flax
import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
import optax
import logging

from flax.training.train_state import TrainState
from gymnasium.spaces import Space
from pathlib import Path
from src.algos.dqn import DQNetwork, EPS_TYPE
from src.utilities.buffers import ReplayBuffer, DictReplayBuffer
from typing import List, Dict, Callable, Tuple
from datetime import datetime
from functools import partial
from jax import jit


# noinspection DuplicatedCode,PyTypeChecker
class SingleModelMADQN(object):
	
	_n_agents: int
	_agent_dqn: DQNetwork
	_replay_buffer: ReplayBuffer
	_write_tensorboard: bool
	_use_vdn: bool
	_use_ddqn: bool
	_use_v2: bool
	
	def __init__(self, num_agents: int, action_dim: int, num_layers: int, act_function: Callable, layer_sizes: List[int], buffer_size: int, gamma: float,
				 action_space: Space, observation_space: Space, use_gpu: bool, dueling_dqn: bool = False, use_ddqn: bool = False, use_vdn: bool = False,
				 use_cnn: bool = False, use_v2: bool = False, handle_timeout: bool = False, use_tensorboard: bool = False, tensorboard_data: List = None,
				 cnn_properties: List[int] = None):
		
		"""
		Initialize a multi-agent scenario DQN with a single DQN model

		:param num_agents: number of agents in the environment
		:param action_dim: number of actions of the agent, the DQN is agnostic to the semantic of each action
        :param num_layers: number of layers for the q_network
        :param act_function: activation function for the q_network
        :param layer_sizes: number of neurons in each layer (list must have equal number of entries as the number of layers)
        :param buffer_size: buffer size for the replay buffer
        :param gamma: reward discount factor
        :param observation_space: gym space for the agent observations
        :param use_gpu: flag that controls use of cpu or gpu
        :param handle_timeout: flag that controls handle timeout termination (due to timelimit) separately and treat the task as infinite horizon task.
        :param use_tensorboard: flag that notes usage of a tensorboard summary writer (default: False)
        :param tensorboard_data: list of the form [log_dir: str, queue_size: int, flush_interval: int, filename_suffix: str] with summary data for
        the summary writer (default is None)

        :type num_agents: int
        :type action_dim: int
        :type num_layers: int
        :type buffer_size: int
        :type layer_sizes: list[int]
        :type use_gpu: bool
        :type handle_timeout: bool
        :type use_tensorboard: bool
        :type gamma: float
        :type act_function: callable
        :type observation_space: gym.Space
        :type tensorboard_data: list
		"""
		
		self._n_agents = num_agents
		self._write_tensorboard = use_tensorboard
		self._use_vdn = use_vdn
		self._use_ddqn = use_ddqn
		self._use_v2 = use_v2
		now = datetime.now()
		if use_tensorboard and tensorboard_data is not None:
			log_name = (tensorboard_data[0] + '/single_model_' + ('vdn_' if use_vdn else '') + now.strftime("%Y%m%d-%H%M%S"))
			if len(tensorboard_data) == 4:
				board_data = [log_name, tensorboard_data[1], tensorboard_data[2], tensorboard_data[3], 'central_train']
			else:
				board_data = [log_name + '_' + tensorboard_data[4], tensorboard_data[1], tensorboard_data[2], tensorboard_data[3], 'central_train']
		else:
			board_data = tensorboard_data
		self._agent_dqn = DQNetwork(action_dim, num_layers, act_function, layer_sizes, gamma, dueling_dqn, use_ddqn, use_cnn, use_tensorboard,
									board_data, cnn_properties, use_v2)
		has_dict_space = isinstance(observation_space, gymnasium.spaces.Dict)
		buffer_type = DictReplayBuffer if has_dict_space else ReplayBuffer
		if use_vdn:
			self._replay_buffer = buffer_type(buffer_size, observation_space, action_space, "cuda" if use_gpu else "cpu",
											   handle_timeout_termination=handle_timeout, n_agents=num_agents)
		else:
			self._replay_buffer = buffer_type(buffer_size * num_agents, observation_space, action_space, "cuda" if use_gpu else "cpu",
											   handle_timeout_termination=handle_timeout)
		
	########################
	### Class Properties ###
	########################
	@property
	def num_agents(self) -> int:
		return self._n_agents
	
	@property
	def agent_dqn(self) -> DQNetwork:
		return self._agent_dqn
	
	@property
	def write_tensorboard(self) -> bool:
		return self._write_tensorboard
	
	@property
	def replay_buffer(self) -> ReplayBuffer:
		return self._replay_buffer
	
	@property
	def use_vdn(self) -> bool:
		return self._use_vdn
	
	#####################
	### Class Methods ###
	#####################
	def mse_loss(self, params: flax.core.FrozenDict, observations: jnp.ndarray, actions: jnp.ndarray, next_q_value: jnp.ndarray):
		q = jnp.zeros((next_q_value.shape[0]))
		for idx in range(self._n_agents):
			qa = self._agent_dqn.q_network.apply(params, observations[:, idx])
			q += qa[np.arange(qa.shape[0]), actions[:, idx].squeeze()]
		q = q.reshape(-1, 1)
		return ((q - next_q_value) ** 2).mean(), q
	
	def l2_v2_loss(self, params: flax.core.FrozenDict, observations_conv: jnp.ndarray, observations_arr: jnp.ndarray, actions: jnp.ndarray,
				   next_q_value: jnp.ndarray):
		q = jnp.zeros((next_q_value.shape[0]))
		for idx in range(self._n_agents):
			qa = self._agent_dqn.q_network.apply(params, observations_conv[idx], observations_arr[idx, :, None])
			q += qa[np.arange(qa.shape[0]), actions[idx].squeeze()]
		q = q.reshape(-1, 1)
		return ((q - next_q_value) ** 2).mean(), q
	
	@partial(jit, static_argnums=(0,))
	def compute_vdn_dqn_loss(self, q_state: TrainState, target_state_params: flax.core.FrozenDict, observations: jnp.ndarray, actions: jnp.ndarray,
							  next_observations: jnp.ndarray, rewards: jnp.ndarray, dones: jnp.ndarray):
		n_obs = len(observations)
		next_q_value = jnp.zeros(n_obs)
		for idx in range(self._n_agents):
			next_q_value += self._agent_dqn.compute_dqn_targets(dones, next_observations[:, idx], rewards[:, idx], target_state_params)
		# next_q_value = next_q_value / n_agents
		
		(loss_value, q_pred), grads = jax.value_and_grad(self.mse_loss, has_aux=True)(q_state.params, observations, actions, next_q_value)
		new_q_state = q_state.apply_gradients(grads=grads)
		return loss_value, q_pred, new_q_state
	
	@partial(jit, static_argnums=(0,))
	def compute_vdn_ddqn_loss(self, q_state: TrainState, target_state_params: flax.core.FrozenDict, observations: jnp.ndarray, actions: jnp.ndarray,
							  next_observations: jnp.ndarray, rewards: jnp.ndarray, dones: jnp.ndarray):
		n_obs = len(observations)
		next_q_value = jnp.zeros((n_obs, 1))
		for idx in range(self._n_agents):
			next_q_value += self._agent_dqn.compute_ddqn_targets(dones, next_observations[:, idx], q_state, rewards[:, idx].reshape(-1, 1),
																 target_state_params)
		# next_q_value = next_q_value / n_agents
		
		(loss_value, q_pred), grads = jax.value_and_grad(self.mse_loss, has_aux=True)(q_state.params, observations, actions, next_q_value)
		new_q_state = q_state.apply_gradients(grads=grads)
		return loss_value, q_pred, new_q_state
	
	@partial(jit, static_argnums=(0,))
	def compute_vdn_v2_loss(self, q_state: TrainState, target_state_params: flax.core.FrozenDict, observations_conv: jnp.ndarray,
							observations_arr: jnp.ndarray, actions: jnp.ndarray, next_observations_conv: jnp.ndarray, next_observations_arr: jnp.ndarray,
							rewards: jnp.ndarray, dones: jnp.ndarray):
		n_obs = len(observations_conv[0])
		next_q_value = jnp.zeros(n_obs)
		for idx in range(self._n_agents):
			next_q_value += self._agent_dqn.compute_v2_targets(dones[idx], next_observations_conv[idx], next_observations_arr[idx],
															   q_state, rewards[idx].reshape(-1, 1), target_state_params)
		(loss_value, q_vals), grads = jax.value_and_grad(self.l2_v2_loss, has_aux=True)(q_state.params, observations_conv, observations_arr, actions,
																						next_q_value)
		new_q_state = q_state.apply_gradients(grads=grads)
		return loss_value, q_vals, new_q_state
	
	def train_dqn(self, env: gymnasium.Env, num_iterations: int, max_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float,
				  final_eps: float, eps_type: str, rng_seed: int, logger: logging.Logger, exploration_decay: float = 0.99, warmup: int = 0, train_freq: int = 1,
				  target_freq: int = 100, tensorboard_frequency: int = 1, use_render: bool = False, cycle: int = 0, greedy_action: bool = True):
		
		rng_gen = np.random.default_rng(rng_seed)
		self._replay_buffer.reseed(rng_seed)
		
		# Setup DQNs for training
		obs, *_ = env.reset()
		if not self._agent_dqn.dqn_initialized:
			if self._agent_dqn.cnn_layer:
				self._agent_dqn.init_network_states(rng_seed, obs[0].reshape((1, *obs[0].shape)), optim_learn_rate)
			else:
				self._agent_dqn.init_network_states(rng_seed, obs[0], optim_learn_rate)
		
		start_time = time.time()
		epoch = 0
		sys.stdout.flush()
		start_record_it = cycle * num_iterations
		start_record_epoch = cycle * max_timesteps
		history = []
		
		for it in range(num_iterations):
			if use_render:
				env.render()
			done = False
			episode_rewards = 0
			episode_q_vals = 0
			episode_start = epoch
			episode_history = []
			logger.info("Iteration %d out of %d" % (it + 1, num_iterations))
			while not done:
				
				# interact with environment
				if eps_type == 'epoch':
					eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, epoch, max_timesteps)
				else:
					eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, it, num_iterations)
				
				explore = rng_gen.random() < eps
				if explore: # Exploration
					actions = np.array(env.action_space.sample())
				else: # Exploitation
					actions = []
					for a_idx in range(self._n_agents):
						# Compute q_values
						if self._agent_dqn.cnn_layer:
							q_values = self._agent_dqn.q_network.apply(self._agent_dqn.online_state.params, obs[a_idx].reshape((1, *obs[a_idx].shape)))[0]
						else:
							q_values = self._agent_dqn.q_network.apply(self._agent_dqn.online_state.params, obs[a_idx])
						# Get action
						if greedy_action:
							action = q_values.argmax()
						else:
							pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
							pol = pol / pol.sum()
							action = rng_gen.choice(range(env.action_space[0].n), p=pol)
						action = jax.device_get(action)
						episode_q_vals += (float(q_values[int(action)]) / self._n_agents)
						actions += [action]
					actions = np.array(actions)
				if not self._agent_dqn.cnn_layer:
					episode_history += [self.get_history_entry(obs, actions)]
				next_obs, rewards, terminated, timeout, infos = env.step(actions)
				if use_render:
					env.render()
				
				if len(rewards) == 1:
					rewards = np.array([rewards] * self._n_agents)
				
				if terminated:
					finished = np.ones(self._n_agents)
				else:
					finished = np.zeros(self._n_agents)
				
				# store new samples
				if self.use_vdn:
					self.replay_buffer.add(obs, next_obs, actions, rewards, finished[0], infos)
					episode_rewards += sum(rewards) / self._n_agents
				else:
					for a_idx in range(self._n_agents):
						self.replay_buffer.add(obs[a_idx], next_obs[a_idx], actions[a_idx], rewards[a_idx], finished[a_idx], infos)
						episode_rewards += (rewards[a_idx] / self._n_agents)
				if self._agent_dqn.use_summary:
					self._agent_dqn.summary_writer.add_scalar("charts/reward", sum(rewards) / self._n_agents, epoch + start_record_epoch)
				obs = next_obs
				
				# update Q-network and target network
				self.update_dqn_models(batch_size, epoch, start_time, target_freq, tau, tensorboard_frequency, train_freq, warmup)
				
				epoch += 1
				sys.stdout.flush()
				
				# Check if iteration is over
				if terminated or timeout:
					if self._write_tensorboard:
						episode_len = epoch - episode_start
						self._agent_dqn.summary_writer.add_scalar("charts/mean_episode_q_vals", episode_q_vals / episode_len, epoch + start_record_epoch)
						self._agent_dqn.summary_writer.add_scalar("charts/episode_return", episode_rewards, it + start_record_it)
						self._agent_dqn.summary_writer.add_scalar("charts/mean_episode_return", episode_rewards / episode_len, it + start_record_it)
						self._agent_dqn.summary_writer.add_scalar("charts/episodic_length", episode_len, it + start_record_it)
						self._agent_dqn.summary_writer.add_scalar("charts/epsilon", eps, it + start_record_it)
					logger.debug("Episode over:\tLength: %d\tEpsilon: %.5f\tReward: %f" % (epoch - episode_start, eps, episode_rewards))
					obs, *_ = env.reset()
					done = True
					history += [episode_history]
					episode_rewards = 0
					episode_start = epoch
		
		return history
	
	def update_dqn_models(self, batch_size: int, epoch: int, start_time: float, target_freq: int, tau: float, tensorboard_frequency: int,
						  train_freq: int, warmup: int):
		if epoch >= warmup:
			if epoch % train_freq == 0:
				data = self._replay_buffer.sample(batch_size)
				
				if self._use_v2:
					obs_conv = []
					obs_array = []
					next_obs_conv = []
					next_obs_array = []
					for idx in range(self._n_agents):
						if isinstance(data[idx].observations, dict):
							obs_conv.append(data[idx].observations['conv'])
							obs_array.append(data[idx].observations['array'])
							next_obs_conv.append(data[idx].next_observations['conv'])
							next_obs_array.append(data[idx].next_observations['array'])
						else:
							obs_conv.append(data[idx].observations[0])
							obs_array.append(data[idx].observations[1])
							next_obs_conv.append(data[idx].next_observations[0])
							next_obs_array.append(data[idx].next_observations[1])
					obs_conv = np.array(obs_conv)
					obs_array = np.array(obs_array)
					next_obs_conv = np.array(next_obs_conv)
					next_obs_array = np.array(next_obs_array)
					
					actions = jnp.array([data[idx].actions for idx in range(self._n_agents)])
					rewards = jnp.array([data[idx].rewards for idx in range(self._n_agents)])
					dones = jnp.array([data[idx].dones for idx in range(self._n_agents)])
					
					if self._use_vdn:
						q_state = self._agent_dqn.online_state
						target_params = self._agent_dqn.target_params
						
						loss, q_pred, self._agent_dqn.online_state = self.compute_vdn_v2_loss(q_state, target_params, obs_conv, obs_array, actions,
																							  next_obs_conv, next_obs_array, rewards, dones)
						
						#  update tensorboard
						if self._agent_dqn.use_summary and epoch % tensorboard_frequency == 0:
							self._agent_dqn.tensorboard_writer.add_scalar("losses/td_loss", jax.device_get(loss), epoch)
							self._agent_dqn.tensorboard_writer.add_scalar("losses/avg_q_values", jax.device_get(q_pred).mean(), epoch)
							self._agent_dqn.tensorboard_writer.add_scalar("charts/SPS", int(epoch / (time.time() - start_time)), epoch)
					else:
						for a_idx in range(self._n_agents):
							self._agent_dqn.update_online_model((obs_conv[a_idx], obs_array[a_idx]), actions[a_idx],
																(next_obs_conv[a_idx], next_obs_array[a_idx]), rewards[a_idx], dones[a_idx],
																epoch, start_time, tensorboard_frequency)
				
				else:
					observations = jnp.array([data[idx].observations for idx in range(self._n_agents)])
					next_observations = jnp.array([data[idx].next_observations for idx in range(self._n_agents)])
					actions = jnp.array([data[idx].actions for idx in range(self._n_agents)])
					rewards = jnp.array([data[idx].rewards for idx in range(self._n_agents)])
					dones = jnp.array([data[idx].dones for idx in range(self._n_agents)])
					if self._use_vdn:
						q_state = self._agent_dqn.online_state
						target_params = self._agent_dqn.target_params
						
						if self._use_ddqn:
							loss, q_pred, self._agent_dqn.online_state = self.compute_vdn_ddqn_loss(q_state, target_params, observations, actions,
																									next_observations, rewards, dones)
						else:
							loss, q_pred, self._agent_dqn.online_state = self.compute_vdn_dqn_loss(q_state, target_params, observations, actions,
																								   next_observations, rewards, dones)
							
							#  update tensorboard
							if self._agent_dqn.use_summary and epoch % tensorboard_frequency == 0:
								self._agent_dqn.tensorboard_writer.add_scalar("losses/td_loss", jax.device_get(loss), epoch)
								self._agent_dqn.tensorboard_writer.add_scalar("losses/avg_q_values", jax.device_get(q_pred).mean(), epoch)
								self._agent_dqn.tensorboard_writer.add_scalar("charts/SPS", int(epoch / (time.time() - start_time)), epoch)
					
					else:
						for a_idx in range(self._n_agents):
							self._agent_dqn.update_online_model(observations[a_idx], actions[a_idx], next_observations[a_idx], rewards[a_idx],
																dones[a_idx], epoch, start_time, tensorboard_frequency)
			
			if epoch % target_freq == 0:
				self._agent_dqn.update_target_model(tau)
	
	def update_model(self, batch_size, epoch, start_time, tensorboard_frequency, logger: logging.Logger):
		data = self._replay_buffer.sample(batch_size)
		observations = data.observations
		next_observations = data.next_observations
		actions = data.actions
		rewards = data.rewards
		dones = data.dones
		train_info = ('epoch: %d \t' % epoch)
		
		if self._use_vdn:
			if self._use_ddqn:
				loss, q_pred, self._agent_dqn.online_state = self.compute_vdn_ddqn_loss(self._agent_dqn.online_state, self._agent_dqn.target_params,
																						observations, actions, next_observations, rewards, dones)
			else:
				loss, q_pred, self._agent_dqn.online_state = self.compute_vdn_dqn_loss(self._agent_dqn.online_state, self._agent_dqn.target_params,
																					   observations, actions, next_observations, rewards, dones)
			
			if self._agent_dqn.use_summary and epoch % tensorboard_frequency == 0:
				self._agent_dqn.tensorboard_writer.add_scalar("losses/td_loss", jax.device_get(loss), epoch)
				self._agent_dqn.tensorboard_writer.add_scalar("losses/avg_q_values", jax.device_get(q_pred).mean(), epoch)
				self._agent_dqn.tensorboard_writer.add_scalar("charts/SPS", int(epoch / (time.time() - start_time)), epoch)
		else:
			loss = self._agent_dqn.update_online_model(observations, actions, next_observations, rewards, dones, epoch, start_time, tensorboard_frequency)
		
		train_info += ('loss: %.7f\t' % loss)
		logger.debug('Train Info: ' + train_info)
		return loss
	
	def save_model(self, filename: str, model_dir: Path, logger: logging.Logger) -> None:
		self._agent_dqn.save_model(filename + '_single_model.model', model_dir, logger)
	
	def load_model(self, filename: str, model_dir: Path, logger: logging.Logger, obs_shape: Tuple) -> None:
		self._agent_dqn.load_model(filename + '_single_model.model', model_dir, logger, obs_shape)
	
	def get_history_entry(self, obs: np.ndarray, actions: List):
		
		entry = []
		for idx in range(self._n_agents):
			entry += [' '.join([str(x) for x in obs[idx]]), str(actions[idx])]
		
		return entry


# noinspection PyTypeChecker,DuplicatedCode
class CentralizedMADQN(object):
	
	_num_agents: int
	_madqn: DQNetwork
	_replay_buffer: ReplayBuffer
	_write_tensorboard: bool
	_use_ddqn: bool
	_use_v2: bool
	_joint_action_converter: Callable
	
	def __init__(self, num_agents: int, action_dim: int, num_layers: int, act_converter: Callable, act_function: Callable, layer_sizes: List[int],
				 buffer_size: int, gamma: float, action_space: Space, observation_space: Space, use_gpu: bool, dueling_dqn: bool = False,
				 use_ddqn: bool = False, use_cnn: bool = False, use_v2: bool = False, handle_timeout: bool = False, use_tensorboard: bool = False,
				 tensorboard_data: List = None, cnn_properties: List[int] = None):
		
		self._num_agents = num_agents
		self._use_v2 = use_v2
		self._write_tensorboard = use_tensorboard
		self._joint_action_converter = act_converter
		now = datetime.now()
		if tensorboard_data is not None:
			board_data = [tensorboard_data[0] + '/centralized_madqn_' + now.strftime("%Y%m%d-%H%M%S"), tensorboard_data[1], tensorboard_data[2],
						  tensorboard_data[3], 'centralized_madqn']
		else:
			board_data = tensorboard_data
		has_dict_space = (isinstance(observation_space, gymnasium.spaces.Dict) or
						  isinstance(observation_space, gymnasium.spaces.Tuple) and isinstance(observation_space[0], gymnasium.spaces.Dict))
		buffer_type = DictReplayBuffer if has_dict_space else ReplayBuffer
		self._madqn = DQNetwork(action_dim ** num_agents, num_layers, act_function, layer_sizes, gamma, dueling_dqn, use_ddqn, use_cnn, use_tensorboard,
								board_data, cnn_properties, use_v2=use_v2, n_obs=num_agents)
		if has_dict_space:
			obs_space = observation_space[0] if isinstance(observation_space, gymnasium.spaces.Tuple) else observation_space
			self._replay_buffer = buffer_type(buffer_size, obs_space, action_space, "cuda" if use_gpu else "cpu",
											  handle_timeout_termination=handle_timeout, n_agents=num_agents)
		else:
			self._replay_buffer = buffer_type(buffer_size, observation_space, action_space, "cuda" if use_gpu else "cpu",
											  handle_timeout_termination=handle_timeout, n_agents=num_agents)
		
	########################
	### Class Properties ###
	########################
	@property
	def num_agents(self) -> int:
		return self._num_agents
	
	@property
	def madqn(self) -> DQNetwork:
		return self._madqn
	
	@property
	def write_tensorboard(self) -> bool:
		return self._write_tensorboard
	
	@property
	def get_joint_action(self) -> Callable:
		return self._joint_action_converter
	
	@property
	def replay_buffer(self) -> ReplayBuffer:
		return self._replay_buffer
		
	#####################
	### Class Methods ###
	#####################
	def train_dqn(self, env: gymnasium.Env, num_iterations: int, max_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float,
				  initial_eps: float, final_eps: float, eps_type: str, rng_seed: int, logger: logging.Logger, num_actions: int, exploration_decay: float = 0.99,
				  warmup: int = 0, train_freq: int = 1, target_freq: int = 100, tensorboard_frequency: int = 1, use_render: bool = False, cycle: int = 0):
		
		rng_gen = np.random.default_rng(rng_seed)
		
		# Setup DQNs for training
		obs, *_ = env.reset()
		self._madqn.init_network_states(rng_seed, obs.reshape((1, *obs.shape)), optim_learn_rate)
		
		start_time = time.time()
		epoch = 0
		sys.stdout.flush()
		start_record_it = cycle * num_iterations
		start_record_epoch = cycle * max_timesteps
		history = []
		
		for it in range(num_iterations):
			if use_render:
				env.render()
			done = False
			episode_rewards = 0
			episode_q_vals = 0
			episode_start = epoch
			episode_history = []
			logger.info("Iteration %d out of %d" % (it + 1, num_iterations))
			while not done:
				
				# interact with environment
				if eps_type == 'epoch':
					eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, epoch, max_timesteps)
				else:
					eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, it, num_iterations)
				if rng_gen.random() < eps:
					joint_action = np.array(rng_gen.choice(range(self._madqn.q_network.action_dim)))
				else:
					q_values = self._madqn.q_network.apply(self._madqn.online_state.params, obs.reshape((1, *obs.shape)))[0]
					joint_action = q_values.argmax(axis=-1)
					joint_action = jax.device_get(joint_action)
					episode_q_vals += float(q_values[int(joint_action)])
				actions = self._joint_action_converter(joint_action, self._num_agents, num_actions)
				next_obs, rewards, terminated, timeout, infos = env.step(actions)
				episode_history += [self.get_history_entry(obs, actions)]
				if use_render:
					env.render()
				
				if len(rewards) == 1:
					rewards = np.array([rewards] * self._num_agents)
				
				if terminated or timeout:
					finished = np.ones(self._num_agents)
				else:
					finished = np.zeros(self._num_agents)
				
				# store new samples
				self._replay_buffer.add(obs, next_obs, joint_action, rewards, finished[0], infos)
				step_reward = sum(rewards) / self._num_agents
				episode_rewards += step_reward
				if self._write_tensorboard:
					self._madqn.summary_writer.add_scalar("charts/reward", step_reward, epoch + start_record_epoch)
				obs = next_obs
				
				# update Q-network and target network
				self.update_dqn_models(batch_size, epoch, start_time, target_freq, tau, tensorboard_frequency, train_freq, warmup)
				
				epoch += 1
				sys.stdout.flush()
				
				# Check if iteration is over
				if terminated or timeout:
					if self._write_tensorboard:
						episode_len = epoch - episode_start
						self._madqn.summary_writer.add_scalar("charts/episodic_q_vals", episode_q_vals / episode_len, epoch + start_record_epoch)
						self._madqn.summary_writer.add_scalar("charts/episodic_return", episode_rewards, it + start_record_it)
						self._madqn.summary_writer.add_scalar("charts/episodic_length", episode_len, it + start_record_it)
						self._madqn.summary_writer.add_scalar("charts/epsilon", eps, it + start_record_it)
					obs, *_ = env.reset()
					done = True
					history += [episode_history]
					episode_rewards = 0
					episode_start = epoch
		
		return history
	
	def update_dqn_models(self, batch_size: int, epoch: int, start_time: float, target_freq: int, tau: float, tensorboard_frequency: int,
						  train_freq: int, warmup: int, n_actions: int):
		
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
					actions = jnp.array([act[0] * n_actions + act[1] for act in data.actions])
					rewards = data.rewards.sum(axis=1)
					dones = data.dones
					self.madqn.update_online_model((obs_conv, obs_array[:, 0]), actions, (next_obs_conv, next_obs_array[:, 0]),
												   rewards, dones, epoch, start_time, tensorboard_frequency)
				
				else:
					observations = data.observations
					next_observations = data.next_observations
					actions = jnp.array([act[0] * n_actions + act[1] for act in data.actions])
					rewards = data.rewards.sum(axis=1)
					dones = data.dones
					self.madqn.update_online_model(observations, actions, next_observations, rewards, dones, epoch, start_time, tensorboard_frequency)
			
			if epoch % target_freq == 0:
				self.madqn.update_target_model(tau)
	
	def save_model(self, filename: str, model_dir: Path, logger: logging.Logger) -> None:
		self._madqn.save_model(filename + '_ctce.model', model_dir, logger)
	
	def load_model(self, filename: str, model_dir: Path, logger: logging.Logger, obs_shape: Tuple) -> None:
		self._madqn.load_model(filename + '_ctce.model', model_dir, logger, obs_shape)
	
	def get_history_entry(self, obs: np.ndarray, actions: List):
		
		entry = []
		for idx in range(self._num_agents):
			entry += [' '.join([str(x) for x in obs[idx]]), str(actions[idx])]
		
		return entry
	
