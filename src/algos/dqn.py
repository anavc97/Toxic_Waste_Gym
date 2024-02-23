#! /usr/bin/env python
import math
import pathlib
import random
import time
import sys
import flax
import gymnasium
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import logging

from algos.q_networks import DuelingQNetwork, CNNDuelingQNetwork, DuelingQNetworkV2
from utilities.buffers import ReplayBuffer
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
from flax.training.train_state import TrainState
# from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from gymnasium.spaces import Discrete, Space, MultiDiscrete
from typing import Callable, List, Union
from pathlib import Path
from termcolor import colored
from functools import partial
from jax import jit

EPS_TYPE = {'linear': 1, 'exp': 2, 'log': 3, 'epoch': 4}

class DQNetwork(object):

    _q_network: nn.Module
    _online_state: TrainState
    _target_state_params: flax.core.FrozenDict
    _replay_buffer: ReplayBuffer
    _tensorboard_writer: SummaryWriter
    _gamma: float
    _use_ddqn: bool
    _use_vdn: bool
    _cnn_layer: bool
    
    def __init__(self, action_dim: int, num_layers: int, act_function: Callable, layer_sizes: List[int], buffer_size: int, gamma: float,
                 observation_space: Space, use_gpu: bool, use_ddqn: bool = False, use_vdn: bool = False, cnn_layer: bool = False,
                 handle_timeout: bool = False, use_tensorboard: bool = False, tensorboard_data: List = None, cnn_properties: List[int] = None,
                 n_agents: int = 1):
    
        """
        Initializes a DQN
        
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
        
        :type action_dim: int
        :type num_layers: int
        :type buffer_size: int
        :type layer_sizes: list[int]
        :type use_gpu: bool
        :type handle_timeout: bool
        :type use_tensorboard: bool
        :type gamma: float
        :type act_function: callable
        :type observation_space: gymnasium.spaces.Space
        :type tensorboard_data: list
        
        """
        
        if cnn_layer:
            if cnn_properties is None:
                cnn_size = 128
                cnn_kernel = (3, 3)
                pool_window = (2, 2)
            else:
                cnn_size = cnn_properties[0]
                cnn_kernel = tuple(cnn_properties[1:3])
                pool_window = tuple(cnn_properties[3:5])
            
            #self._q_network = CNNDuelingQNetwork(action_dim=action_dim, num_layers=num_layers, activation_function=act_function,
                                                 #layer_sizes=layer_sizes.copy(), cnn_size=cnn_size, cnn_kernel=cnn_kernel, pool_window=pool_window)
            self._q_network = DuelingQNetworkV2(action_dim=action_dim, num_layers=num_layers, activation_function=act_function,
                                                 layer_sizes=layer_sizes.copy(), cnn_size=cnn_size, cnn_kernel=cnn_kernel, pool_window=pool_window, use_cnn=True)
        else:
            self._q_network = DuelingQNetwork(action_dim=action_dim, num_layers=num_layers, activation_function=act_function, layer_sizes=layer_sizes.copy())
        
        if use_vdn and n_agents > 1:
            self._replay_buffer = ReplayBuffer(buffer_size, observation_space, MultiDiscrete([action_dim] * n_agents), "cuda" if use_gpu else "cpu",
                                               n_agents=n_agents, handle_timeout_termination=handle_timeout)
        else:
            self._replay_buffer = ReplayBuffer(buffer_size, observation_space, Discrete(action_dim), "cuda" if use_gpu else "cpu",
                                               n_agents=n_agents, handle_timeout_termination=handle_timeout)
        
        self._gamma = gamma
        self._use_tensorboard = use_tensorboard
        self._target_state_params = None
        self._online_state = None
        self._use_ddqn = use_ddqn
        self._use_vdn = use_vdn
        self._cnn_layer = cnn_layer
        self._q_network.apply = jax.jit(self._q_network.apply)
        self._dqn_initialized = False
        if use_tensorboard:
            summary_log = tensorboard_data[0]
            queue_size = int(tensorboard_data[1])
            flush_time = int(tensorboard_data[2])
            file_suffix = tensorboard_data[3]
            comment = tensorboard_data[4]
            self._tensorboard_writer = SummaryWriter(log_dir=summary_log, comment=comment, max_queue=queue_size, flush_secs=flush_time,
                                                     filename_suffix=file_suffix)

    #############################
    ##    GETTERS & SETTERS    ##
    #############################

    @property
    def q_network(self) -> nn.Module:
        return self._q_network
    
    @property
    def online_state(self) -> TrainState:
        return self._online_state
    
    @property
    def target_params(self) -> flax.core.FrozenDict:
        return self._target_state_params
    
    @property
    def replay_buffer(self) -> ReplayBuffer:
        return self._replay_buffer
    
    @property
    def summary_writer(self) -> SummaryWriter:
        return self._tensorboard_writer
    
    @property
    def gamma(self) -> float:
        return self._gamma
    
    @property
    def use_summary(self) -> bool:
        return self._use_tensorboard
    
    @property
    def tensorboard_writer(self) -> SummaryWriter:
        return self._tensorboard_writer
    
    @property
    def cnn_layer(self) -> bool:
        return self._cnn_layer
    
    @property
    def use_vdn(self) -> bool:
        return self._use_vdn
    
    @property
    def dqn_initialized(self) -> bool:
        return self._dqn_initialized
    
    @dqn_initialized.setter
    def dqn_initialized(self, new_val: bool) -> None:
        self._dqn_initialized = new_val
    
    @gamma.setter
    def gamma(self, new_gamma: float) -> None:
        self._gamma = new_gamma
        
    @target_params.setter
    def target_params(self, new_params: flax.core.FrozenDict) -> None:
        self._target_state_params = new_params
        
    @online_state.setter
    def online_state(self, new_state: TrainState) -> None:
        self._online_state = new_state

    #############################
    ##       CLASS UTILS       ##
    #############################

    def init_network_states(self, rng_seed: int, obs: np.ndarray, optim_learn_rate: float):
        key = jax.random.PRNGKey(rng_seed)
        key, q_key = jax.random.split(key, 2)
        if self._online_state is None:
            self._online_state = TrainState.create(
                apply_fn=self._q_network.apply,
                params=self._q_network.init(q_key, obs),
                tx=optax.adam(learning_rate=optim_learn_rate),
            )
        if self._target_state_params is None:
            self._target_state_params = self._q_network.init(q_key, obs)
            update_target_state_params = optax.incremental_update(self._online_state.params, self._target_state_params, 1.0)
            self._target_state_params = flax.core.freeze(update_target_state_params)
        
        self._dqn_initialized = True
    
    def compute_dqn_targets(self, dones, next_observations, rewards, target_state_params) -> Union[np.ndarray, jax.Array]:
        q_next_target = self._q_network.apply(target_state_params, next_observations)  # get target network q values
        q_next_target = jnp.max(q_next_target, axis=-1)  # get best q_val for each obs
        return rewards + (1 - dones) * self._gamma * q_next_target  # compute Bellman equation
    
    def compute_ddqn_targets(self, dones, next_observations, q_state, rewards, target_state_params) -> Union[np.ndarray, jax.Array]:
        q_next_target = self._q_network.apply(target_state_params, next_observations)  # get target network q values
        # DDQN the target network is used to evaluate how good the online network's predictions are
        q_next_online = self._q_network.apply(q_state.params, next_observations)  # get online network's prescribed actions
        online_acts = jnp.argmax(q_next_online, axis=1)
        q_next_target = q_next_target[np.arange(q_next_target.shape[0]), online_acts.squeeze()].reshape(-1, 1)  # get target's q values for prescribed actions
        # print(q_next_target.shape)
        return rewards + (1 - dones) * self._gamma * q_next_target  # compute Bellman equation
    
    def mse_loss(self, params: flax.core.FrozenDict, observations: Union[np.ndarray, jax.Array], actions: Union[np.ndarray, jax.Array],
                     next_q_value: Union[np.ndarray, jax.Array]):
        q = self._q_network.apply(params, observations)  # get online model's q_values
        q = q[np.arange(q.shape[0]), actions.squeeze()].reshape(-1, 1)
        return ((q - next_q_value) ** 2).mean(), q  # compute loss
    
    def mse_vdn_loss(self, params: flax.core.FrozenDict, observations: Union[np.ndarray, jax.Array], actions: Union[np.ndarray, jax.Array],
                     next_q_value: Union[np.ndarray, jax.Array], n_agents:int, n_obs: int):
        q = jnp.zeros(n_obs)
        for idx in range(n_agents):
            qa = self._q_network.apply(params, observations[:, idx])  # get online model's q_values
            # print(qa[np.arange(qa.shape[0]), actions[:, idx].squeeze()].shape)
            q += qa[np.arange(qa.shape[0]), actions[:, idx].squeeze()]
        # q = q / n_agents
        q = q.reshape(-1, 1)
        
        return ((q - next_q_value) ** 2).mean(), q  # compute loss

    @partial(jit, static_argnums=(0,))
    def compute_dqn_loss(self, q_state: TrainState, target_state_params: flax.core.FrozenDict, observations: Union[np.ndarray, jax.Array],
                         actions: Union[np.ndarray, jax.Array], next_observations: Union[np.ndarray, jax.Array], rewards: Union[np.ndarray, jax.Array],
                         dones: Union[np.ndarray, jax.Array]):
        next_q_value = self.compute_dqn_targets(dones, next_observations, rewards, target_state_params)
        
        (loss_value, q_pred), grads = jax.value_and_grad(self.mse_loss, has_aux=True)(q_state.params, observations, actions, next_q_value)
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, q_pred, q_state
    
    @partial(jit, static_argnums=(0,))
    def compute_ddqn_loss(self, q_state: TrainState, target_state_params: flax.core.FrozenDict, observations: Union[np.ndarray, jax.Array],
                          actions: Union[np.ndarray, jax.Array], next_observations: Union[np.ndarray, jax.Array], rewards: Union[np.ndarray, jax.Array],
                          dones: Union[np.ndarray, jax.Array]):
        next_q_value = self.compute_ddqn_targets(dones, next_observations, q_state, rewards, target_state_params)
        
        (loss_value, q_pred), grads = jax.value_and_grad(self.mse_loss, has_aux=True)(q_state.params, observations, actions, next_q_value)
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, q_pred, q_state
    
    @partial(jit, static_argnums=(0,))
    def compute_vdn_dqn_loss(self, q_state: TrainState, target_state_params: flax.core.FrozenDict, observations: Union[np.ndarray, jax.Array],
                             actions: Union[np.ndarray, jax.Array], next_observations: Union[np.ndarray, jax.Array], rewards: Union[np.ndarray, jax.Array],
                             dones: Union[np.ndarray, jax.Array]):
        n_obs, n_agents, *_ = observations.shape
        next_q_value = np.zeros(n_obs)
        for idx in range(n_agents):
            next_q_value += self.compute_dqn_targets(dones, next_observations[:, idx], rewards[:, idx], target_state_params)
        # next_q_value = next_q_value / n_agents
        
        (loss_value, q_pred), grads = jax.value_and_grad(self.mse_vdn_loss, has_aux=True)(q_state.params, observations, actions, next_q_value, n_agents, n_obs)
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, q_pred, q_state
    
    @partial(jit, static_argnums=(0,))
    def compute_vdn_ddqn_loss(self, q_state: TrainState, target_state_params: flax.core.FrozenDict, observations: Union[np.ndarray, jax.Array],
                             actions: Union[np.ndarray, jax.Array], next_observations: Union[np.ndarray, jax.Array], rewards: Union[np.ndarray, jax.Array],
                             dones: Union[np.ndarray, jax.Array]):
        n_obs, n_agents, *_ = observations.shape
        next_q_value = np.zeros((n_obs, 1))
        for idx in range(n_agents):
            next_q_value += self.compute_ddqn_targets(dones, next_observations[:, idx], q_state, rewards[:, idx].reshape(-1, 1), target_state_params)
        # next_q_value = next_q_value / n_agents
        
        (loss_value, q_pred), grads = jax.value_and_grad(self.mse_vdn_loss, has_aux=True)(q_state.params, observations, actions, next_q_value, n_agents, n_obs)
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, q_pred, q_state
    
    @staticmethod
    def eps_update(update_type: int, init_eps: float, end_eps: float, decay_rate: float, step: int, max_steps: int):
        
        if update_type == 1:
            return max(((end_eps - init_eps) / max_steps) * step / decay_rate + init_eps, end_eps)
        elif update_type == 2:
            return max(decay_rate ** step * init_eps, end_eps)
        elif update_type == 3:
            return max((1 / (1 + decay_rate * step)) * init_eps, end_eps)
        elif update_type == 4:
            return max((decay_rate * math.sqrt(step)) * init_eps, end_eps)
        else:
            print(colored('Unrecognized exploration decay type, defaulting to logarithmic decay', 'red'))
            return max((1 / (1 + decay_rate * step)) * init_eps, end_eps)

    def train(self, env: gymnasium.Env, num_iterations: int, max_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float,
              final_eps: float, eps_type: str, rng_seed: int, exploration_decay: float = 0.99, warmup: int = 0, target_freq: int = 1000, train_freq: int = 10,
              summary_frequency: int = 1):

        obs, _ = env.reset()
        rng_gen = np.random.default_rng(rng_seed)
        self.init_network_states(rng_seed, obs, optim_learn_rate)

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
                    q_values = self._q_network.apply(self._online_state.params, obs)
                    action = q_values.argmax(axis=-1)
                    action = jax.device_get(action)
                next_obs, reward, finished, timeout, info = env.step(action)
                episode_rewards += reward
                episode_history += [obs, action]
                
                # store new samples
                real_next_obs = next_obs.copy()
                self._replay_buffer.add(obs, real_next_obs, action, reward, np.array(finished), info)
                obs = next_obs
        
                # update Q-network and target network
                if epoch > warmup:
                    if epoch % train_freq == 0:
                        self.update_online_model(batch_size, epoch, start_time, summary_frequency)
                    
                    if epoch % target_freq == 0:
                        self.update_target_model(tau)
    
                epoch += 1
                sys.stdout.flush()
                if finished:
                    obs, _ = env.reset()
                    done = True
                    history += [episode_history]
                    if self._use_tensorboard:
                        self._tensorboard_writer.add_scalar("charts/episodic_return", episode_rewards, epoch)
                        self._tensorboard_writer.add_scalar("charts/episodic_length", epoch - episode_start, epoch)
                        self._tensorboard_writer.add_scalar("charts/epsilon", eps, epoch)
                        print("Episode over:\tReward: %f\tLength: %d" % (episode_rewards, epoch - episode_start))
        
        return history
    
    def train_cnn(self, env: gymnasium.Env, num_iterations: int, max_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float,
                  final_eps: float, eps_type: str, rng_seed: int, logger: logging.Logger, exploration_decay: float = 0.99, warmup: int = 0,
                  target_freq: int = 1000, train_freq: int = 10, summary_frequency: int = 1):

        env.reset()
        obs = env.render()
        self.init_network_states(rng_seed, obs, optim_learn_rate)

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
                if random.random() < eps:
                    action = np.array(env.action_space.sample())
                else:
                    q_values = self._q_network.apply(self._online_state.params, obs)
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
                if epoch > warmup:
                    if epoch % train_freq == 0:
                        self.update_online_model(batch_size, epoch, start_time, summary_frequency)
                    
                    if epoch % target_freq == 0:
                        self.update_target_model(tau)
    
                epoch += 1
                sys.stdout.flush()
                if finished:
                    env.reset()
                    obs = env.render()
                    done = True
                    history += [episode_history]
                    if self._use_tensorboard:
                        self._tensorboard_writer.add_scalar("charts/episodic_return", episode_rewards, epoch)
                        self._tensorboard_writer.add_scalar("charts/episodic_length", epoch - episode_start, epoch)
                        self._tensorboard_writer.add_scalar("charts/epsilon", eps, epoch)
                        logger.debug("Episode over:\tReward: %f\tLength: %d" % (episode_rewards, epoch - episode_start))
        
        return history
    
    def update_online_model(self, batch_size: int, epoch: int, start_time: float, summary_frequency: int) -> float:
        data = self._replay_buffer.sample(batch_size)
        
        observations = data.observations
        actions = data.actions
        next_observations = data.next_observations
        rewards = data.rewards
        finished = data.dones
        
        # perform a gradient-descent step
        if self._use_ddqn:
            if self._use_vdn:
                td_loss, q_val, self._online_state = self.compute_vdn_ddqn_loss(self._online_state, self._target_state_params, observations, actions,
                                                                                next_observations, rewards, finished)
            else:
                td_loss, q_val, self._online_state = self.compute_ddqn_loss(self._online_state, self._target_state_params, observations, actions,
                                                                            next_observations, rewards, finished)
        else:
            if self._use_vdn:
                td_loss, q_val, self._online_state = self.compute_vdn_dqn_loss(self._online_state, self._target_state_params, observations, actions,
                                                                               next_observations, rewards, finished)
            else:
                td_loss, q_val, self._online_state = self.compute_dqn_loss(self._online_state, self._target_state_params, observations, actions,
                                                                           next_observations, rewards, finished)
        
        #  update tensorboard
        if self._use_tensorboard and epoch % summary_frequency == 0:
            self._tensorboard_writer.add_scalar("losses/td_loss", jax.device_get(td_loss), epoch)
            self._tensorboard_writer.add_scalar("losses/avg_q_values", jax.device_get(q_val).mean(), epoch)
            self._tensorboard_writer.add_scalar("charts/SPS", int(epoch / (time.time() - start_time)), epoch)
            # print("Loss: %.5f\tQ_value: %.5f" % (float(jax.device_get(td_loss)), float(jax.device_get(q_val).mean())))
        return td_loss
    
    def update_target_model(self, tau: float):
        update_target_state_params = optax.incremental_update(self._online_state.params, self._target_state_params.unfreeze(), tau)
        self._target_state_params = flax.core.freeze(update_target_state_params)
    
    def get_action(self, obs):
        q_values = self._q_network.apply(self._q_network.variables, obs)
        actions = q_values.argmax()
        return jax.device_get(actions)
    
    def create_checkpoint(self, model_dir: Path, epoch: int = 0) -> None:
        save_checkpoint(ckpt_dir=model_dir, target=self._online_state, step=epoch)
    
    def load_checkpoint(self, ckpt_file: Path, logger: logging.Logger, epoch: int = -1) -> None:
        template = TrainState.create(apply_fn=self._q_network.apply,
                                     params=self._q_network.init(jax.random.PRNGKey(201), jnp.empty((1, 7))),
                                     tx=optax.adam(learning_rate=0.0001))
        if epoch < 0:
            if pathlib.Path.is_file(ckpt_file):
                self._online_state = restore_checkpoint(ckpt_dir=ckpt_file, target=template)
            else:
                logger.error('ERROR!! Could not load checkpoint, expected checkpoint file got directory instead')
        else:
            if pathlib.Path.is_dir(ckpt_file):
                self._online_state = restore_checkpoint(ckpt_dir=ckpt_file, target=template, step=epoch)
            else:
                logger.error('ERROR!! Could not load checkpoint, expected checkpoint directory got file instead')
    
    def save_model(self, filename: str, model_dir: Path, logger: logging.Logger) -> None:
        file_path = model_dir / (filename + '.model')
        with open(file_path, "wb") as f:
            f.write(flax.serialization.to_bytes(self._online_state))
        logger.info("Model state saved to file: " + str(file_path))
    
    def load_model(self, filename: str, model_dir: Path, logger: logging.Logger, obs_shape: tuple) -> None:
        file_path = model_dir / filename
        template = TrainState.create(apply_fn=self._q_network.apply,
                                     params=self._q_network.init(jax.random.PRNGKey(201), jnp.empty(obs_shape)),
                                     tx=optax.adam(learning_rate=0.0001))
        with open(file_path, "rb") as f:
            self._online_state = flax.serialization.from_bytes(template, f.read())
        logger.info("Loaded model state from file: " + str(file_path))
