#! /usr/bin/env python

import flax.linen as nn
import jax.numpy as jnp
from typing import Callable, List, Tuple


class DuelingQNetwork(nn.Module):
	action_dim: int
	num_layers: int
	layer_sizes: List[int]
	activation_function: Callable
	
	@nn.compact
	def __call__(self, x_orig: jnp.ndarray):
		x = jnp.array(x_orig)
		for i in range(self.num_layers):
			x = self.activation_function(nn.Dense(self.layer_sizes[i])(x))
		a = nn.Dense(self.action_dim)(x)
		v = nn.Dense(1)(x)
		return v + (a - a.mean())


class CNNDuelingQNetwork(nn.Module):
	action_dim: int
	num_layers: int
	layer_sizes: List[int]
	activation_function: Callable
	cnn_size: int
	cnn_kernel: Tuple[int]
	pool_window: Tuple[int]
	
	@nn.compact
	def __call__(self, x_orig: jnp.ndarray):
		x = self.activation_function(nn.Conv(self.cnn_size, kernel_size=self.cnn_kernel)(x_orig))
		x = nn.avg_pool(x, window_shape=self.pool_window)
		x = x.reshape((x.shape[0], -1))
		for i in range(self.num_layers):
			x = self.activation_function(nn.Dense(self.layer_sizes[i])(x))
		a = nn.Dense(self.action_dim)(x)
		v = nn.Dense(1)(x)
		return v + (a - a.mean())
		# return v + (a - a.max())


class DuelingQNetworkV2(nn.Module):
	action_dim: int
	num_layers: int
	layer_sizes: List[int]
	activation_function: Callable
	cnn_size: int
	cnn_kernel: Tuple[int]
	pool_window: Tuple[int]
	use_cnn: bool
	
	@nn.compact
	def __call__(self, x_orig: jnp.ndarray):
		if self.use_cnn:
			x_shape = x_orig.shape
			x = self.activation_function(nn.Conv(self.cnn_size, kernel_size=self.cnn_kernel)(x_orig[0]))
			x = nn.avg_pool(x, window_shape=self.pool_window)
			x = x.reshape((x.shape[0], -1))
			x = jnp.stack([x, x_orig[1]])
		else:
			x = x_orig
		for i in range(self.num_layers):
			x = self.activation_function(nn.Dense(self.layer_sizes[i])(x))
		a = nn.Dense(self.action_dim)(x)
		v = nn.Dense(1)(x)
		return v + (a - a.mean())