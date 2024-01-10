#! /usr/bin/env python

import sys
import numpy as np
import gym
import pickle
import yaml
import itertools
import random
import time
import os
import copy
import logging

from pathlib import Path
from typing import List, Tuple, Callable, Dict, Any
from time import time
from threading import Lock, Thread
from queue import Queue, LifoQueue, Empty, Full
from src.env.astro_waste_env import AstroWasteEnv, Actions
from collections import namedtuple
from abc import ABC, abstractmethod


START_POINTS = 100


class GeneralAstroWasteGame(ABC):
	Response = namedtuple("Response", ["result", "reason"])
	
	_players: Dict[int, Tuple[str, int]]	# players in the game
	_player_actions: Dict[int, int]			# next player action by id
	_game_id: int							# game id for referencing
	_is_active: bool						# flag signaling if game is running or paused
	_cycles_second: int						# number of game cycles per second
	_maximum_players: int					# maximum number of players allowed
	_levels: List[str]						# list with the levels' names to be played
	_game_env: AstroWasteEnv				# game environment
	_points: float							# game points
	_time_spent: float						# time since game started
	_cycles_run: int						# cycles run since game started
	
	def __init__(self, cycles_second: int, levels: List[str], max_players: int = 2, game_id: int = 0):
		
		self._cycles_second = cycles_second
		self._levels = levels.copy()
		self._maximum_players = max_players
		self._game_id = game_id
		
		self._players = {}
		self._player_actions = {}
		self._is_active = False
		self._time_spent = 0.0
		self._cycles_run = 0
		self._points = START_POINTS
		self._game_env = None

	###########################
	### GETTERS AND SETTERS ###
	###########################
	
	@property
	def players(self) -> Dict:
		return self._players
	
	@property
	def player_actions(self) -> Dict:
		return self._player_actions
	
	@property
	def levels(self) -> List[str]:
		return self._levels.copy()
	
	@property
	def game_id(self) -> int:
		return self._game_id
	
	@property
	def max_players(self) -> int:
		return self._maximum_players
	
	@property
	def time_spent(self) -> float:
		return self._time_spent
		
	@property
	def cycles_run(self) -> int:
		return self._cycles_run
	
	@property
	def points(self) -> float:
		return self._points
	
	@property
	def is_active(self) -> bool:
		return self._is_active
	
	@property
	def game_env(self) -> AstroWasteEnv:
		return self._game_env
	
	@levels.setter
	def levels(self, new_levels: List[str]):
		self._levels = new_levels.copy()
		
	@max_players.setter
	def max_players(self, new_max: int):
		self._maximum_players = new_max
		
	@points.setter
	def points(self, new_points: float):
		self._points = new_points
		
	@cycles_run.setter
	def cycles_run(self, new_cycles: int):
		self._cycles_run = new_cycles
		
	@time_spent.setter
	def time_spent(self, new_time: float):
		self._time_spent = new_time
	
	#################################
	### GAME MANAGEMENT UTILITIES ###
	#################################
	
	def enque_action(self, agent_id: int, action: int) -> Response:
		"""
		Sets next action for given player to execute
		
		:param agent_id: string with the identification of the agent to execute action
		:param action: integer identifying action to perform from Action enum
		:return: response: Response tuple with the result of enqueuing action
		"""
		player_ids = list(self._player_actions.keys())
		if agent_id in player_ids:
			if action in iter(Actions):
				self._player_actions[agent_id] = action
				return self.Response(result='no_error', reason='')
			else:
				return self.Response(result='action_error', reason='Action %d not amongst valid actions' % action)
		else:
			return self.Response(result='id_error', reason='No playing agent with id %d' % agent_id)
		
	def get_state(self) -> Dict:
		"""
		
		:return:
		"""
		state = {}
		env_state = self._game_env.create_observation()
		state['players'] = env_state.players
		state['objects'] = env_state.objects
		state['finished'] = env_state.game_finished
		state['timeout'] = env_state.game_timeout
		return state
	
	def env_step(self) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
		actions = []
		env_agents = self._game_env.players
		
		for agent_idx in range(self._game_env.n_players):
			actions += [self._player_actions[env_agents[agent_idx].id]]
		
		for player_id in self._player_actions.keys():
			self._player_actions[player_id] = Actions.STAY
		
		return self._game_env.step(actions)
	
	def add_player(self, player_id: int, player_name: str, player_type: int) -> Response:
		n_players = len(self._players)
		if n_players < self._maximum_players:
			if player_id not in self._players.keys():
				self._players[player_id] = (player_name, player_type)
				return self.Response(result='no_error', reason='')
			else:
				return self.Response(result='id_exists_error', reason='Player with id %d already exists' % player_id)
		else:
			return self.Response(result='max_players_error', reason='Maximum number of players already reached, cannot add more')
	
	def remove_player(self, player_id: int) -> Response:
		if player_id in self._players.keys():
			self._players.pop(player_id)
			return self.Response(result='no_error', reason='')
		else:
			return self.Response(result='no_id_exists_error', reason='Player with id %d does not exist' % player_id)
	
	def pause_game(self) -> Response:
		if self._is_active:
			self._is_active = False
			return self.Response(result='no_error', reason='')
		else:
			return self.Response(result='game_paused_error', reason='Game already paused, cannot be paused')
	
	def unpause_game(self) -> Response:
		if not self._is_active:
			self._is_active = True
			return self.Response(result='no_error', reason='')
		else:
			return self.Response(result='game_running_error', reason='Game is already running, cannot be put running again')
	
	def is_full(self) -> bool:
		return int(len(self._players)) >= self._maximum_players
	
	def is_finished(self) -> bool:
		return self._game_env.is_over()
	
	def env_reset(self, seed=None) -> None:
		self._game_env.reset(seed=seed)
	
	def get_game_metadata(self) -> Dict:
		metadata = self.get_state()
		metadata['game_id'] = self._game_id
		metadata['points'] = self._points
		metadata['game_time'] = self._time_spent
		metadata['ticks'] = self._cycles_run
		return metadata
	
	#########################
	### GAME MAIN METHODS ###
	#########################
	
	@abstractmethod
	def game_main_loop(self) -> None:
		raise NotImplementedError("Specific game approaches should implement the main loop behaviour")
	
	
class PythonAstroGame(GeneralAstroWasteGame):
	
	def game_main_loop(self) -> None:
		pass