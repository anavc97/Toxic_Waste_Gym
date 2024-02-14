#! /usr/bin/env python

import numpy as np

from typing import List, Tuple, Dict, Any
from src.env.toxic_waste_env_v2 import ToxicWasteEnvV2, Actions
from collections import namedtuple


START_POINTS = 100


class AstroWasteGame(object):
	Response = namedtuple("Response", ["result", "reason"])
	
	_players: Dict[int, Tuple[str, int]]	# players in the game
	_player_actions: Dict[int, int]			# next player action by id
	_game_id: int							# game id for referencing
	_is_active: bool						# flag signaling if game is running or paused
	_game_started: bool						# flag signaling if game has started or not
	_cycles_second: int						# number of game cycles per second
	_maximum_players: int					# maximum number of players allowed
	_levels: List[str]						# list with the levels' names to be played
	_game_env: ToxicWasteEnvV2				# game environment
	_points: float							# game points
	_time_spent: float						# time since game started
	_cycles_run: int						# cycles run since game started
	_lvl_idx: int							# current level idx
	
	def __init__(self, cycles_second: int, levels: List[str], game_env: ToxicWasteEnvV2, max_players: int = 2, game_id: int = 0):
		
		self._cycles_second = cycles_second
		self._levels = levels.copy()
		self._maximum_players = max_players
		self._game_id = game_id
		self._game_started = False
		
		self._players = {}
		self._player_actions = {}
		self._is_active = False
		self._time_spent = 0.0
		self._cycles_run = 0
		self._lvl_idx = 0
		self._points = START_POINTS
		self._game_env = game_env

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
	def game_env(self) -> ToxicWasteEnvV2:
		return self._game_env
	
	@property
	def level_idx(self) -> int:
		return self._lvl_idx
	
	@property
	def game_started(self) -> bool:
		return self._game_started
	
	@levels.setter
	def levels(self, new_levels: List[str]) -> None:
		self._levels = new_levels.copy()
		
	@max_players.setter
	def max_players(self, new_max: int) -> None:
		self._maximum_players = new_max
		
	@points.setter
	def points(self, new_points: float) -> None:
		self._points = new_points
		
	@cycles_run.setter
	def cycles_run(self, new_cycles: int) -> None:
		self._cycles_run = new_cycles
		
	@time_spent.setter
	def time_spent(self, new_time: float) -> None:
		self._time_spent = new_time
	
	@game_env.setter
	def game_env(self, new_env: ToxicWasteEnvV2) -> None:
		self._game_env = new_env
	
	@level_idx.setter
	def level_idx(self, new_idx: int) -> None:
		self._lvl_idx = new_idx
	
	def add_level(self, new_lvl: str) -> None:
		self._levels.append(new_lvl)
	
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
		state['players'] = [player.to_dict() for player in self._game_env.players]
		state['objects'] = [obj.to_dict() for obj in self._game_env.objects]
		state['finished'] = env_state.game_finished
		state['timeout'] = env_state.game_timeout
		state['score'] = env_state.score
		state['time_left'] = env_state.time_left
		return state
	
	def env_step(self) -> tuple:
		actions = []
		env_agents = self._game_env.players
		
		for agent_idx in range(self._game_env.n_players):
			actions += [self._player_actions[env_agents[agent_idx].id]]
		
		for player_id in self._player_actions.keys():
			self._player_actions[player_id] = Actions.STAY
		
		obs, _, _, _, info = self._game_env.step(actions)
		
		return obs, info, self._player_actions
	
	def add_player(self, player_id: int, player_name: str, player_type: int) -> Response:
		n_players = len(self._players)
		if n_players < self._maximum_players:
			if player_id not in self._players.keys():
				self._players[player_id] = (player_name, player_type)
				self._player_actions[player_id] = Actions.STAY
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
	
	def level_finished(self) -> bool:
		# TODO: Update to finish when human player goes through "door"
		return self._game_env.is_over()
	
	def game_finished(self) -> bool:
		return self.level_finished() and self._lvl_idx >= (len(self._levels) - 1)
	
	def start_game(self) -> None:
		self._game_started = True
	
	def env_reset(self, seed=None) -> tuple[np.ndarray, dict[str, Any]]:
		self._game_env.layout = self._levels[self._lvl_idx]
		return self._game_env.reset(seed=seed)
	
	def get_game_metadata(self) -> Dict:
		metadata = self.get_state()
		metadata['game_id'] = self._game_id
		# metadata['points'] = self._points
		metadata['game_time'] = self._time_spent
		metadata['ticks'] = self._cycles_run
		return metadata
	
	