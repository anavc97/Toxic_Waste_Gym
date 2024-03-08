#! /usr/bin/env python

import numpy as np
import yaml
import gymnasium
import time

from env.toxic_waste_env_base import BaseToxicEnv, AgentType, HoldState, WasteState, PlayerState, CellEntity
from pathlib import Path
from enum import IntEnum, Enum
from gymnasium.spaces import Discrete, Box
from typing import List, Tuple, Any, Union
from termcolor import colored
from collections import namedtuple
from copy import deepcopy


MOVE_REWARD = 0.0
HOLD_REWARD = -1.0
DELIVER_WASTE = 10
ROOM_CLEAN = 50
IDENTIFY_REWARD = 1


class Actions(IntEnum):
	UP = 0
	DOWN = 1
	LEFT = 2
	RIGHT = 3
	INTERACT = 4
	STAY = 5
	IDENTIFY = 6


class ActionDirection(Enum):
	UP = (-1, 0)
	DOWN = (1, 0)
	LEFT = (0, -1)
	RIGHT = (0, 1)
	INTERACT = (0, 0)
	STAY = (0, 0)
	IDENTIFY = (0, 0)


class WasteType(IntEnum):
	GREEN = 1
	YELLOW = 2
	RED = 3


class WasteStateV2(WasteState):
	_points: float
	_time_penalty: float
	_identified: bool
	_waste_type: int
	
	def __init__(self, position: Tuple[int, int], obj_id: str, points: float = 1, time_penalty: float = 0.0, hold_state: int = HoldState.FREE.value,
				 waste_type: int = WasteType.GREEN, holding_player: 'PlayerState' = None, identified: bool = False):
		super().__init__(position, obj_id, hold_state, holding_player)
		self._points = points
		self._time_penalty = time_penalty
		self._identified = identified
		self._waste_type = waste_type
	
	@property
	def identified(self) -> bool:
		return self._identified
	
	@property
	def waste_type(self) -> int:
		return self._waste_type
	
	@property
	def points(self) -> float:
		return self._points
	
	@property
	def time_penalty(self) -> float:
		return self._time_penalty
	
	@identified.setter
	def identified(self, new_val: bool) -> None:
		self._identified = new_val
		
	def deepcopy(self):
		new_obj = WasteStateV2(self._position, self._id, self._points, self._time_penalty, identified=self._identified)
		new_obj.hold_state = self._hold_state
		return new_obj
	
	def __eq__(self, other):
		return isinstance(other, WasteStateV2) and self._id == other._id and self._position == other._position
	
	def __repr__(self):
		return ("%s@(%d, %d), held_status: %s, identified? %r" %
				(self._id, self._position[0], self._position[1], HoldState(self._hold_state).name, self._identified))
	
	def to_dict(self):
		return {"name": self._id, "position": self._position, "hold_state": self._hold_state, "identified": self._identified, "type": self._waste_type,
				"holding_player": self._holding_player.id if self._holding_player else None}
	
	@classmethod
	def from_dict(cls, obj_dict):
		obj_dict = deepcopy(obj_dict)
		return WasteStateV2(**obj_dict)


# noinspection PyUnresolvedReferences
class ToxicWasteEnvV2(BaseToxicEnv):
	"""
	Collaborative game environment of toxic waste collection, useful for ad-hoc teamwork research.
	
	Version 2 - the agents have a fixed timelimit to collect all the waste and exist different types of waste that can have different impacts on the players'
	scoring and time remaining. Also, to help with identifying different wastes, the autonomous agent has access to an extra action of identification of waste.
	"""
	Observation = namedtuple("Observation",
							 ["field", "players", "objects", "game_finished", "game_timeout", "sight", "current_step", "time_left", "time_penalties",
							  "score"])
	
	def __init__(self, terrain_size: Tuple[int, int], layout: str, max_players: int, max_objects: int, max_steps: int, rnd_seed: int,
				 require_facing: bool = False, agent_centered: bool = False, render_mode: List[str] = None, slip: bool = False, is_train: bool = False,
				 dict_obs: bool = True):

		self._dict_obs = dict_obs
		self._is_train = is_train
		self._slip = slip
		self._slip_prob = 0.0
		self._max_time = 0.0
		self._time_penalties = 0.0
		self._score = 0.0
		self._door_pos = (-1, 1)
		super().__init__(terrain_size, layout, max_players, max_objects, max_steps, rnd_seed, 'v2', require_facing, True, agent_centered,
						 False, render_mode)
		self._start_time = time.time()
	
	###########################
	### GETTERS AND SETTERS ###
	###########################
	@property
	def slip(self) -> bool:
		return self._slip
	
	@slip.setter
	def slip(self, new_val: bool) -> None:
		self._slip = new_val
	
	@property
	def score(self) -> float:
		return self._score
	
	@property
	def door_pos(self) -> Tuple:
		return self._door_pos
	
	#######################
	### UTILITY METHODS ###
	#######################
	def add_object(self, position: Tuple, obj_id: str = 'ball', points: int = 1, time_penalty: float = 1, waste_type: int = WasteType.GREEN) -> bool:
		
		if self._n_objects < self._max_objects:
			self._objects.append(WasteStateV2(position, obj_id, points=points, time_penalty=time_penalty, identified=False, waste_type=waste_type))
			self._n_objects += 1
			return True
		else:
			print(colored('[ADD_OBJECT] Max number of objects (%d) already reached, cannot add a new one.' % self._max_objects, 'yellow'))
			return False
	
	def _get_observation_space(self) -> Union[gymnasium.spaces.Tuple, gymnasium.spaces.Dict]:
		
		if self._agent_centered_obs:
			# grid observation space
			grid_shape = (1 + 2 * self._agent_sight, 1 + 2 * self._agent_sight)
			
			# agents layer: agent levels
			agents_min = np.zeros(grid_shape, dtype=np.int32)
			agents_max = np.ones(grid_shape, dtype=np.int32)
			
			# waste layer: waste pos
			green_min = np.zeros(grid_shape, dtype=np.int32)
			green_max = np.ones(grid_shape, dtype=np.int32)
			yellow_min = np.zeros(grid_shape, dtype=np.int32)
			yellow_max = np.ones(grid_shape, dtype=np.int32)
			red_min = np.zeros(grid_shape, dtype=np.int32)
			red_max = np.ones(grid_shape, dtype=np.int32)
			
			# access layer: i the cell available
			occupancy_min = np.zeros(grid_shape, dtype=np.int32)
			occupancy_max = np.ones(grid_shape, dtype=np.int32)
			
			# total layer
			min_obs = np.stack([agents_min, green_min, yellow_min, red_min, occupancy_min])
			max_obs = np.stack([agents_max, green_max, yellow_max, red_max, occupancy_max])
		
		else:
			# grid observation space
			grid_shape = (self._rows, self._cols)
			
			# agents layer
			agents_min = np.zeros(grid_shape, dtype=np.int32)
			agents_max = np.ones(grid_shape, dtype=np.int32)
			
			# objects layer
			green_min = np.zeros(grid_shape, dtype=np.int32)
			green_max = np.ones(grid_shape, dtype=np.int32)
			yellow_min = np.zeros(grid_shape, dtype=np.int32)
			yellow_max = np.ones(grid_shape, dtype=np.int32)
			red_min = np.zeros(grid_shape, dtype=np.int32)
			red_max = np.ones(grid_shape, dtype=np.int32)
			
			# occupancy layer
			occupancy_min = np.zeros(grid_shape, dtype=np.int32)
			occupancy_max = np.ones(grid_shape, dtype=np.int32)
			
			# acting agent layer
			acting_agent_min = np.zeros(grid_shape, dtype=np.int32)
			acting_agent_max = np.ones(grid_shape, dtype=np.int32)
			
			# total layer
			min_obs = np.stack([agents_min, green_min, yellow_min, red_min, occupancy_min, acting_agent_min])
			max_obs = np.stack([agents_max, green_max, yellow_max, red_max, occupancy_max, acting_agent_max])
		
		if self._dict_obs:
			return gymnasium.spaces.Dict({'conv': Box(np.array(min_obs), np.array(max_obs), dtype=np.int32),
										  'array': Box(np.array(0), np.array(self.max_steps), dtype=np.float32)})
		else:
			return gymnasium.spaces.Tuple([Box(np.array(min_obs), np.array(max_obs), dtype=np.int32),
										   Box(np.array(0), np.array(self.max_steps), dtype=np.float32)])
	
	def setup_env(self) -> None:
		
		config_filepath = Path(__file__).parent.absolute() / 'data' / 'configs' / 'layouts' / (self._room_layout + '.yaml')
		with open(config_filepath) as config_file:
			config_data = yaml.safe_load(config_file)
		field_data = config_data['field']
		players_data = config_data['players']
		objects_data = config_data['objects']
		self._slip_prob = float(config_data['slip_prob'])
		self._max_time = float(config_data['max_train_time']) if self._is_train else float(config_data['max_game_time'])
		config_sight = float(config_data['sight'])
		self._agent_sight = config_sight if config_sight > 0 else min(self._rows, self._cols)
		n_red = 0
		n_green = 0
		n_yellow = 0
		
		for row in range(self._rows):
			for col in range(self._cols):
				cell_val = field_data[row][col]
				if cell_val == ' ':
					pass
				elif cell_val == 'X':
					self._field[row, col] = CellEntity.COUNTER
				elif cell_val == 'T':
					self._field[row, col] = CellEntity.TOXIC
				elif cell_val == 'I':
					self._field[row, col] = CellEntity.ICE
				elif cell_val == 'D':
					self._field[row, col] = CellEntity.DOOR
					self._door_pos = (row, col)
				elif cell_val == 'G':
					self.add_object((row, col), objects_data['green']['ids'][n_green], objects_data['green']['points'],
									objects_data['green']['time_penalty'], waste_type=WasteType.GREEN)
					self._field[row, col] = CellEntity.COUNTER
					n_green += 1
				elif cell_val == 'R':
					self.add_object((row, col), objects_data['red']['ids'][n_red], objects_data['red']['points'],
									objects_data['red']['time_penalty'], waste_type=WasteType.RED)
					self._field[row, col] = CellEntity.COUNTER
					n_red += 1
				elif cell_val == 'Y':
					self.add_object((row, col), objects_data['yellow']['ids'][n_yellow], objects_data['yellow']['points'],
									objects_data['yellow']['time_penalty'], waste_type=WasteType.YELLOW)
					self._field[row, col] = CellEntity.COUNTER
					n_yellow += 1
				elif cell_val.isdigit():
					nxt_player_data = players_data[self._n_players]
					self.add_player((row, col), tuple(nxt_player_data['orientation']), nxt_player_data['id'], nxt_player_data['name'],
									AgentType[nxt_player_data['type'].upper()].value)
				else:
					print(colored("[SETUP_ENV] Cell value %s not recognized, considering empty cell" % cell_val, 'yellow'))
					continue
	
	def is_game_finished(self) -> bool:
		return any([self._field[p.position[0], p.position[1]] == CellEntity.DOOR for p in self.players if p.agent_type == AgentType.HUMAN])
	
	def is_game_timedout(self) -> bool:
		return self.get_time_left() <= 0
	
	def move_ice(self, move_agent: PlayerState, next_position: Tuple) -> Tuple:
		
		agent_pos = move_agent.position
		right_move = (next_position[0] - agent_pos[0], next_position[1] - agent_pos[1])
		wrong_moves = [direction.value for direction in ActionDirection if direction.value != right_move and direction.value != (0, 0)]
		n_wrong_moves = len(wrong_moves)
		moves_prob = np.array([1 - self._slip_prob] + [self._slip_prob / n_wrong_moves] * n_wrong_moves)
		possible_positions = ([next_position] + [(max(min(wrong_move[0] + agent_pos[0], self._rows), 0), max(min(wrong_move[1] + agent_pos[1], self.cols), 0))
												 for wrong_move in wrong_moves])
		return possible_positions[self._np_random.choice(range(len(possible_positions)), p=moves_prob)]
	
	def get_time_left(self) -> float:
		if not self._is_train:
			curr_time = self._max_time - (time.time() - self._start_time)
		else:
			curr_time = self.max_steps - self._current_step
		
		return curr_time - self._time_penalties
	
	def get_object_facing(self, player: PlayerState) -> WasteStateV2:
		facing_pos = (player.position[0] + player.orientation[0], player.position[1] + player.orientation[1])
		for obj in self._objects:
			if obj.position == facing_pos and obj.hold_state == HoldState.FREE:
				return obj
		
		return None
	
	####################
	### MAIN METHODS ###
	####################
	def create_observation(self) -> Observation:
		return self.Observation(field=self.field,
								players=self.players,
								objects=self.objects,
								game_finished=self.is_game_finished(),
								game_timeout=self.is_game_timedout(),
								sight=self._agent_sight,
								current_step=self._current_step,
								time_left=self.get_time_left(),
								time_penalties=self._time_penalties,
								score=self._score)
	
	def get_env_log(self) -> str:
		
		env_log = 'Environment state:\nPlayer states:\n'
		for player in self._players:
			env_log += '\t- ' + str(player) + '\n'
		
		env_log += 'Object states:\n'
		for obj in self._objects:
			if obj.hold_state != HoldState.HELD:
				env_log += '\t- ' + str(obj) + '\n'
		
		env_log += 'Field layout: %s\n' % str(self._field)
		env_log += 'Current timestep: %d\nGame is finished: %r\n' % (self._current_step, self.is_game_finished())
		env_log += 'Game has timed out: %r\nTime left: %f' % (self.is_game_timedout(), self.get_time_left())
		
		return env_log
	
	def make_obs_grid(self) -> Union[np.ndarray, List]:
		
		if self._agent_centered_obs:
			layers_size = (self._rows + 2 * self._agent_sight, self._cols + 2 * self._agent_sight)
			agent_layer = np.zeros(layers_size, dtype=np.int32)
			green_layer = np.zeros(layers_size, dtype=np.int32)
			yellow_layer = np.zeros(layers_size, dtype=np.int32)
			red_layer = np.zeros(layers_size, dtype=np.int32)
			occupancy_layer = np.ones(layers_size, dtype=np.int32)
			occupancy_layer[:self._agent_sight, :] = 0
			occupancy_layer[-self._agent_sight:, :] = 0
			occupancy_layer[:, :self._agent_sight] = 0
			occupancy_layer[:, -self._agent_sight:] = 0
			
			for agent in self._players:
				pos = agent.position
				agent_layer[pos[0] + self._agent_sight, pos[1] + self._agent_sight] = 1
				occupancy_layer[pos[0] + self._agent_sight, pos[1] + self._agent_sight] = 0
				
			for obj in self._objects:
				pos = obj.position
				if obj.waste_type == WasteType.GREEN:
					green_layer[pos[0] + self._agent_sight, pos[1] + self._agent_sight] = 1
				elif obj.waste_type == WasteType.YELLOW:
					yellow_layer[pos[0] + self._agent_sight, pos[1] + self._agent_sight] = 1
				elif obj.waste_type == WasteType.RED:
					red_layer[pos[0] + self._agent_sight, pos[1] + self._agent_sight] = 1
				occupancy_layer[pos[0] + self._agent_sight, pos[1] + self._agent_sight] = 0
			
			for row in range(self._rows):
				for col in range(self._cols):
					if self._field[row, col] == CellEntity.COUNTER:
						occupancy_layer[row + self._agent_sight, col + self._agent_sight] = 0
			
			obs = np.stack([agent_layer, green_layer, yellow_layer, red_layer, occupancy_layer])
			padding = 2 * self._agent_sight + 1
			time_left = self.get_time_left()
			
			if self._dict_obs:
				return [{'conv': obs[:, a.position[0]:a.position[0] + padding, a.position[1]:a.position[1] + padding], 'array': np.array(time_left)}
						for a in self._players]
			else:
				return np.array([np.array([obs[:, a.position[0]:a.position[0] + padding, a.position[1]:a.position[1] + padding], np.array(time_left)],
										  dtype=object)
								 for a in self._players])
		
		else:
			layers_size = (self._rows, self._cols)
			agent_layer = np.zeros(layers_size, dtype=np.int32)
			green_layer = np.zeros(layers_size, dtype=np.int32)
			yellow_layer = np.zeros(layers_size, dtype=np.int32)
			red_layer = np.zeros(layers_size, dtype=np.int32)
			occupancy_layer = np.ones(layers_size, dtype=np.int32)
			acting_layer = np.zeros((self._n_players, *layers_size), dtype=np.int32)
			
			for agent_idx in range(self._n_players):
				pos = self._players[agent_idx].position
				agent_layer[pos[0], pos[1]] = 1
				occupancy_layer[pos[0], pos[1]] = 0
				acting_layer[agent_idx, pos[0], pos[1]] = 1
			
			for obj in self._objects:
				pos = obj.position
				if obj.waste_type == WasteType.GREEN:
					green_layer[pos[0], pos[1]] = 1
				elif obj.waste_type == WasteType.YELLOW:
					yellow_layer[pos[0], pos[1]] = 1
				elif obj.waste_type == WasteType.RED:
					red_layer[pos[0], pos[1]] = 1
				occupancy_layer[pos[0], pos[1]] = 0
			
			for row in range(self._rows):
				for col in range(self._cols):
					if self._field[row, col] == CellEntity.COUNTER:
						occupancy_layer[row, col] = 0
			time_left = self.get_time_left()
			
			if self._dict_obs:
				return [{'conv': np.stack([agent_layer, green_layer, yellow_layer, red_layer, occupancy_layer, acting_layer[idx]]), 'array': np.array(time_left)}
						for idx in range(self._n_players)]
			else:
				return np.array([np.array([np.stack([agent_layer, green_layer, yellow_layer, red_layer, occupancy_layer, acting_layer[idx]]),
										   np.array(time_left)],
										  dtype=object)
								 for idx in range(self._n_players)])
	
	def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
		
		obs, info = super().reset(seed=seed, options=options)
		self._start_time = time.time()
		
		return obs, info
	
	def step(self, actions: List[int]) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
		
		slip_agents = self.execute_transitions(actions)
		finished = self.is_game_finished()
		rewards = np.array([player.reward for player in self._players])
		timeout = self.is_game_timedout()
		return self.make_obs(), rewards, finished, timeout, {'agents_slipped': slip_agents}
	
	def execute_transitions(self, actions: List[int]) -> List[int]:
		
		self._current_step += 1
		old_positions = []
		for player in self._players:
			player.reward = MOVE_REWARD
			old_positions += [player.position]
		
		# Process action list
		new_positions = []
		slip_agents = []
		agents_disposed_waste = []
		waste_disposed = {}
		for act_idx in range(self.n_players):
			act = actions[act_idx]
			acting_player = self._players[act_idx]
			act_direction = ActionDirection[Actions(act).name].value
			if act != Actions.INTERACT and act != Actions.STAY and act != Actions.IDENTIFY:
				acting_player.orientation = act_direction
			next_pos = (max(min(acting_player.position[0] + act_direction[0], self._rows), 0),
						max(min(acting_player.position[1] + act_direction[1], self.cols), 0))
			if not (self._field[next_pos] == CellEntity.DOOR or self._field[next_pos] == CellEntity.EMPTY or
					self._field[next_pos] == CellEntity.TOXIC or self._field[next_pos] == CellEntity.ICE):
				new_positions.append(acting_player.position)
			elif self._slip and self._field[acting_player.position] == CellEntity.ICE:
				new_positions.append(self.move_ice(acting_player, next_pos))
				slip_agents.append(acting_player.id)
			else:
				new_positions.append(next_pos)
			
			# Handle INTERACT action is only necessary for human agents
			if act == Actions.INTERACT and acting_player.agent_type == AgentType.HUMAN:
				facing_pos = (acting_player.position[0] + acting_player.orientation[0], acting_player.position[1] + acting_player.orientation[1])
				agent_facing = self.get_agent_facing(acting_player)
				if acting_player.is_holding_object():
					if agent_facing is not None:  # facing an agent
						agent_idx = self._players.index(agent_facing)
						agent_action = actions[agent_idx]
						agent_type = agent_facing.agent_type
						if agent_type == AgentType.ROBOT  and (
								agent_action == Actions.STAY or agent_action == Actions.INTERACT):  # check if the agent is a robot and is not trying to move
							if self.require_facing and not self.are_facing(acting_player, agent_facing):
								continue
							# Place object in robot
							place_obj = acting_player.held_objects[0]
							acting_player.drop_object(place_obj.id)
							place_obj.hold_state = HoldState.DISPOSED
							place_obj.identified = True
							place_obj.holding_player = agent_facing
							place_obj.position = (-1, -1)
							agent_facing.hold_object(place_obj)
							agents_disposed_waste.append(acting_player)
							agents_disposed_waste.append(agent_facing)
							waste_disposed[acting_player.id] = place_obj.points
							if place_obj.waste_type == WasteType.RED:
								waste_disposed[agent_facing.id] = 0
							else:
								waste_disposed[agent_facing.id] = place_obj.points
							self._score += place_obj.points
					else:
						# Drop object to the field
						dropped_obj = acting_player.held_objects[0]
						if dropped_obj.hold_state == HoldState.HELD and self.free_pos(facing_pos):
							acting_player.drop_object(dropped_obj.id)
							dropped_obj.position = facing_pos
							dropped_obj.hold_state = HoldState.FREE
							dropped_obj.holding_player = None
				
				else:
					if agent_facing is None:
						# Pick object from counter or floor
						for obj in self._objects:
							if obj.position == facing_pos and obj.hold_state == HoldState.FREE:
								pick_obj = obj
								pick_obj.position = acting_player.position
								pick_obj.hold_state = HoldState.HELD
								pick_obj.holding_player = acting_player
								#pick_obj.identified = True
								acting_player.hold_object(pick_obj)
								# self._time_penalties += pick_obj.time_penalty			# Uncomment if it is supposed to apply penalty at pickup
			
			# IDENTIFY action only has impact by robot agents
			elif act == Actions.IDENTIFY and acting_player.agent_type == AgentType.ROBOT:
				object_facing = self.get_object_facing(acting_player)
				acting_player.reward = IDENTIFY_REWARD
				if not object_facing.identified:
					object_facing.identified = True
		
		# Handle movement and collisions
		# Check for collisions (more than one agent moving to same position or two agents switching position)
		can_move = []
		for idx in range(self._n_players):
			curr_pos = old_positions[idx]
			next_pos = new_positions[idx]
			add_move = True
			for idx2 in range(self._n_players):
				if idx2 == idx:
					continue
				if ((next_pos == old_positions[idx2] and old_positions[idx2] == new_positions[idx2]) or new_positions[idx2] == next_pos or
						(curr_pos == new_positions[idx2] and next_pos == old_positions[idx2])):
					add_move = False
					break
			if add_move:
				can_move.append(idx)
		# Update position for agents with valid movements
		for idx in can_move:
			moving_player = self._players[idx]
			old_pos = old_positions[idx]
			next_pos = new_positions[idx]
			ball_pos = [obj.position for obj in self.objects]
			if idx == AgentType.HUMAN and moving_player.is_holding_object():
				self._time_penalties += sum([obj.time_penalty for obj in moving_player.held_objects])	# When the agent moves holding waste apply time penalty
				self._players[AgentType.ROBOT].reward = HOLD_REWARD
			if old_pos != next_pos and next_pos not in ball_pos:
				moving_player.position = next_pos
				if moving_player.is_holding_object():
					for obj in moving_player.held_objects:
						if obj.hold_state != HoldState.DISPOSED:
							obj.position = next_pos
		
		for player in self._players:
			if self.is_game_finished():
				player.reward = ROOM_CLEAN
			elif player in agents_disposed_waste:
				player.reward = waste_disposed[player.id]
			else:
				facing_agent = self.get_agent_facing(player)
				if (facing_agent is not None and
						((player.agent_type == AgentType.HUMAN and facing_agent.agent_type == AgentType.ROBOT and player.is_holding_object()) or
						 (player.agent_type == AgentType.ROBOT and facing_agent.agent_type == AgentType.HUMAN and facing_agent.is_holding_object()))):
					player.reward = DELIVER_WASTE / self._max_time_steps
		
		return slip_agents
	
	def render(self) -> np.ndarray | list[np.ndarray] | None:
		if self._render is None:
			try:
				from .render import Viewer
				self._render = Viewer((self.rows, self.cols), visible=self._show_viewer)
			except Exception as e:
				print('Caught exception %s when trying to import Viewer class.' % str(e.args))
		
		return self._render.render(self, return_rgb_array=(self.render_mode == 'rgb_array'))