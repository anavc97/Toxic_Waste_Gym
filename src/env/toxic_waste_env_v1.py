#! /usr/bin/env python

import numpy as np
import yaml
import gymnasium

from src.env.toxic_waste_env_base import BaseToxicEnv, ObjectState
from pathlib import Path
from enum import IntEnum, Enum
from gymnasium.utils import seeding
from gymnasium.spaces import Discrete, Box
from gymnasium import Env
from typing import List, Tuple, Any
from copy import deepcopy
from termcolor import colored
from collections import namedtuple


MOVE_REWARD = 0.0
HOLD_REWARD = -3.0
DELIVER_WASTE = 10
ROOM_CLEAN = 50


class AgentType(IntEnum):
	HUMAN = 0
	ROBOT = 1


class HoldState(IntEnum):
	FREE = 0
	HELD = 1
	DISPOSED = 2


class CellEntity(IntEnum):
	EMPTY = 0
	COUNTER = 1
	TOXIC = 2
	ICE = 3
	AGENT = 4
	OBJECT = 5


class Actions(IntEnum):
	UP = 0
	DOWN = 1
	LEFT = 2
	RIGHT = 3
	INTERACT = 4
	STAY = 5


class ActionDirection(Enum):
	UP = (-1, 0)
	DOWN = (1, 0)
	LEFT = (0, -1)
	RIGHT = (0, 1)
	INTERACT = (0, 0)
	STAY = (0, 0)


class PlayerState(object):
	_position: Tuple[int, int]
	_orientation: Tuple[int, int]
	_name: str
	_id: int
	_agent_type: int
	_held_object: List[ObjectState]
	_reward: float
	
	def __init__(self, pos: Tuple[int, int], orientation: Tuple[int, int], agent_id: int, agent_name: str, agent_type: int,
				 held_object: List[ObjectState] = None):
		self._position = pos
		self._orientation = orientation
		self._agent_type = agent_type
		self._name = agent_name
		self._id = agent_id
		self._held_object = held_object
		self._reward = 0
		
		if self._held_object is not None:
			for obj in self._held_object:
				assert isinstance(obj, ObjectState)
				assert obj.position == self._position
	
	@property
	def position(self) -> Tuple:
		return self._position
	
	@property
	def orientation(self) -> Tuple:
		return self._orientation
	
	@property
	def agent_type(self) -> int:
		return self._agent_type
	
	@property
	def id(self) -> int:
		return self._id
	
	@property
	def name(self) -> str:
		return self._name
	
	@property
	def reward(self) -> float:
		return self._reward
	
	@property
	def held_objects(self) -> List[ObjectState]:
		if self._held_object is not None:
			return self._held_object.copy()
		else:
			return self._held_object

	@position.setter
	def position(self, new_pos: Tuple[int, int]) -> None:
		self._position = new_pos
		
	@orientation.setter
	def orientation(self, new_orientation: Tuple[int, int]) -> None:
		self._orientation = new_orientation
	
	@reward.setter
	def reward(self, new_val: float) -> None:
		self._reward = new_val
	
	def hold_object(self, other_obj: ObjectState) -> None:
		assert isinstance(other_obj, ObjectState), "[HOLD OBJECT ERROR] object is not an ObjectState"
		if self._held_object is not None:
			self._held_object.append(other_obj)
		else:
			self._held_object = [other_obj]
			
	def drop_object(self, obj_id: str) -> None:
		assert self.is_holding_object(), "[DROP OBJECT] holding no objects"
		
		for obj in self._held_object:
			if obj.id == obj_id:
				self._held_object.remove(obj)
				return
		
		print(colored('[DROP OBJECT] no object found with id %s' % obj_id, 'yellow'))

	def is_holding_object(self) -> bool:
		if self._held_object is not None:
			return len(self._held_object) > 0
		else:
			return False

	def deepcopy(self):
		held_objs = (None if self._held_object is None else [self._held_object[idx].deepcopy() for idx in range(len(self._held_object))])
		return PlayerState(self._position, self._orientation, self._id, self._name, self._agent_type, held_objs)
	
	def __eq__(self, other):
		return (
				isinstance(other, PlayerState)
				and self.position == other.position
				and self.orientation == other.orientation
				and self.held_objects == other.held_objects
		)
	
	def __hash__(self):
		return hash((self.position, self.orientation, self.held_objects))
	
	def __repr__(self):
		return "Agent {} at {} facing {} holding {}".format(self._name, self.position, self.orientation, str(self.held_objects))
	
	def to_dict(self):
		return {
			"position": self.position,
			"orientation": self.orientation,
			"held_object": [self.held_objects[idx].to_dict() for idx in range(len(self._held_object))] if self.held_objects is not None else None,
		}
	
	@staticmethod
	def from_dict(player_dict):
		player_dict = deepcopy(player_dict)
		held_obj = player_dict.get("held_object", None)
		if held_obj is not None:
			player_dict["held_object"] = [ObjectState.from_dict(held_obj[idx]) for idx in range(len(held_obj))]
		return PlayerState(**player_dict)


# noinspection PyUnresolvedReferences
class AstroWasteEnv(BaseToxicEnv):
	Observation = namedtuple("Observation", ["field", "players", "objects", "game_finished", "game_timeout", "sight", "current_step"])
	
	def __init__(self, terrain_size: Tuple[int, int], layout: str, max_players: int, max_objects: int, max_steps: int, rnd_seed: int,
				 require_facing: bool = False, layer_obs: bool = False, agent_centered: bool = False, use_encoding: bool = False,
				 render_mode: List[str] = None, slip: bool = False):
		
		super().__init__(terrain_size, layout, max_players, max_objects, max_steps, rnd_seed, require_facing, layer_obs, agent_centered,
						 use_encoding, render_mode)
		self._slip = slip
		self._slip_prob = 0.0
	
	###########################
	### GETTERS AND SETTERS ###
	###########################
	@property
	def slip(self) -> bool:
		return self._slip
	
	@slip.setter
	def slip(self, new_val: bool) -> None:
		self._slip = new_val
	
	#######################
	### UTILITY METHODS ###
	#######################
	def _get_observation_space(self) -> Box:
		
		if self._use_layer_obs:
			if self._agent_centered_obs:
				# grid observation space
				grid_shape = (1 + 2 * self._agent_sight, 1 + 2 * self._agent_sight)
				
				# agents layer: agent levels
				agents_min = np.zeros(grid_shape, dtype=np.int32)
				agents_max = np.ones(grid_shape, dtype=np.int32)
				
				# foods layer: foods level
				objs_min = np.zeros(grid_shape, dtype=np.int32)
				objs_max = np.ones(grid_shape, dtype=np.int32)
				
				# access layer: i the cell available
				occupancy_min = np.zeros(grid_shape, dtype=np.int32)
				occupancy_max = np.ones(grid_shape, dtype=np.int32)
				
				# total layer
				min_obs = np.stack([agents_min, objs_min, occupancy_min])
				max_obs = np.stack([agents_max, objs_max, occupancy_max])
			
			else:
				# grid observation space
				grid_shape = (self._rows, self._cols)
				
				# agents layer
				agents_min = np.zeros(grid_shape, dtype=np.int32)
				agents_max = np.ones(grid_shape, dtype=np.int32)
				
				# objects layer
				objs_min = np.zeros(grid_shape, dtype=np.int32)
				objs_max = np.ones(grid_shape, dtype=np.int32)
				
				# occupancy layer
				occupancy_min = np.zeros(grid_shape, dtype=np.int32)
				occupancy_max = np.ones(grid_shape, dtype=np.int32)
				
				# acting agent layer
				acting_agent_min = np.zeros(grid_shape, dtype=np.int32)
				acting_agent_max = np.ones(grid_shape, dtype=np.int32)
				
				# total layer
				min_obs = np.stack([agents_min, objs_min, occupancy_min, acting_agent_min])
				max_obs = np.stack([agents_max, objs_max, occupancy_max, acting_agent_max])
		else:
			if self._use_encoding:
				min_obs = [-1, -1, 0, 0, 0] * self._n_players + [-1, -1, *[0] * len(HoldState), *[0] * self._n_players] * self._n_objects
				max_obs = ([self._rows - 1, self._cols - 1, 1, 1, 1] * self._n_players +
						   [self._rows - 1, self._cols - 1, *[1] * len(HoldState), *[1] * self._n_players] * self._n_objects)
			else:
				min_obs = [-1, -1, -1, -1, 0] * self._n_players + [-1, -1, 0, 0] * self._n_objects
				max_obs = [self._rows - 1, self._cols - 1, 1, 1, 1] * self._n_players + [self._rows - 1, self._cols - 1, 2, self._n_players - 1] * self._n_objects
		
		return Box(np.array(min_obs), np.array(max_obs), dtype=np.int32)
	
	def setup_env(self) -> None:
		
		config_filepath = Path(__file__).parent.absolute() / 'data' / 'configs' / 'layouts' / (self._room_layout + '.yaml')
		with open(config_filepath) as config_file:
			config_data = yaml.safe_load(config_file)
		field_data = config_data['field']
		players_data = config_data['players']
		objects_data = config_data['objects']
		self._slip_prob = float(config_data['slip_prob'])
		config_sight = float(config_data['sight'])
		self._agent_sight = config_sight if config_sight > 0 else min(self._rows, self._cols)
		
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
				elif cell_val == 'O':
					self.add_object((row, col), objects_data[self._n_objects])
					self._field[row, col] = CellEntity.COUNTER
				elif cell_val.isdigit():
					nxt_player_data = players_data[self._n_players]
					self.add_player((row, col), tuple(nxt_player_data['orientation']), nxt_player_data['id'], nxt_player_data['name'],
									AgentType[nxt_player_data['type'].upper()].value)
				else:
					print(colored("[SETUP_ENV] Cell value %s not recognized, considering empty cell" % cell_val, 'yellow'))
					continue
	
	@staticmethod
	def encode_orientation(orientation: Tuple) -> Tuple[int, int]:
		if orientation == ActionDirection.UP.value:
			return 0, 0
		elif orientation == ActionDirection.DOWN.value:
			return 0, 1
		elif orientation == ActionDirection.LEFT.value:
			return 1, 0
		elif orientation == ActionDirection.RIGHT.value:
			return 1, 1
		else:
			return 0, 0
	
	def is_game_finished(self) -> bool:
		return (self._n_objects - len(self.disposed_objects())) == 0
	
	def move_ice(self, move_agent: PlayerState, next_position: Tuple) -> Tuple:
		
		agent_pos = move_agent.position
		right_move = (next_position[0] - agent_pos[0], next_position[1] - agent_pos[1])
		wrong_moves = [direction.value for direction in ActionDirection if direction.value != right_move and direction.value != (0, 0)]
		n_wrong_moves = len(wrong_moves)
		moves_prob = np.array([1 - self._slip_prob] + [self._slip_prob / n_wrong_moves] * n_wrong_moves)
		possible_positions = ([next_position] + [(max(min(wrong_move[0] + agent_pos[0], self._rows), 0), max(min(wrong_move[1] + agent_pos[1], self.cols), 0))
												 for wrong_move in wrong_moves])
		return possible_positions[self._np_random.choice(range(len(possible_positions)), p=moves_prob)]
	
	####################
	### MAIN METHODS ###
	####################
	def get_env_log(self) -> str:
		
		env_log = 'Environment state:\nPlayer states:\n'
		for player in self._players:
			env_log += '\t- ' + str(player) + '\n'
		
		env_log += 'Object states:\n'
		for obj in self._objects:
			if obj.hold_state != HoldState.HELD:
				env_log += '\t- ' + str(obj) + '\n'
		
		env_log += 'Field layout: %s\n' % str(self._field)
		env_log += 'Current timestep: %d\nGame is finished: %r\nGame has timed out: %r\n' % (self._current_step, self.is_game_finished(), self.is_game_timedout())
		
		return env_log
	
	def make_obs_array(self) -> np.ndarray:
		state = []
		
		for player in self._players:
			player_state = [player.position, player.orientation, int(player.is_holding_object())]
			for other_player in self._players:
				if other_player.id != player.id:
					player_state.append(other_player.position)
					player_state.append(other_player.orientation)
					player_state.append(int(other_player.is_holding_object()))
			for obj in self._objects:
				if obj.hold_state == HoldState.DISPOSED:  # When disposed, object position is a virtual trash bin at (-1, -1)
					player_state.append((-1, -1))
				else:
					player_state.append(obj.position)
				player_state.append(obj.hold_state)
				if obj.hold_state == HoldState.FREE:  # If the object is not held, the agent holding it is 0 (None)
					player_state.append(0)
				else:
					player_state.append(self._players.index(obj.holding_player) + 1)
			
			state.append(player_state)
		
		return np.array(state)
	
	def make_obs_grid(self) -> np.ndarray:
		
		if self._agent_centered_obs:
			layers_size = (self._rows + 2 * self._agent_sight, self._cols + 2 * self._agent_sight)
			agent_layer = np.zeros(layers_size, dtype=np.int32)
			obj_layer = np.zeros(layers_size, dtype=np.int32)
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
				obj_layer[pos[0] + self._agent_sight, pos[1] + self._agent_sight] = 1
				occupancy_layer[pos[0] + self._agent_sight, pos[1] + self._agent_sight] = 0
			
			for row in range(self._rows):
				for col in range(self._cols):
					if self._field[row, col] == CellEntity.COUNTER:
						occupancy_layer[row + self._agent_sight, col + self._agent_sight] = 0
			
			obs = np.stack([agent_layer, obj_layer, occupancy_layer])
			padding = 2 * self._agent_sight + 1
			
			return np.array([obs[:, a.position[0]:a.position[0] + padding, a.position[1]:a.position[1] + padding] for a in self._players])
		
		else:
			layers_size = (self._rows, self._cols)
			agent_layer = np.zeros(layers_size, dtype=np.int32)
			obj_layer = np.zeros(layers_size, dtype=np.int32)
			occupancy_layer = np.ones(layers_size, dtype=np.int32)
			acting_layer = np.zeros((self._n_players, *layers_size), dtype=np.int32)
			
			for agent_idx in range(self._n_players):
				pos = self._players[agent_idx].position
				agent_layer[pos[0], pos[1]] = 1
				occupancy_layer[pos[0], pos[1]] = 0
				acting_layer[agent_idx, pos[0], pos[1]] = 1
			
			for obj in self._objects:
				pos = obj.position
				obj_layer[pos[0], pos[1]] = 1
				occupancy_layer[pos[0], pos[1]] = 0
			
			for row in range(self._rows):
				for col in range(self._cols):
					if self._field[row, col] == CellEntity.COUNTER:
						occupancy_layer[row, col] = 0
			
			return np.array([np.stack([agent_layer, obj_layer, occupancy_layer, acting_layer[idx]]) for idx in range(self._n_players)])
	
	def make_obs_dqn(self) -> np.ndarray:
		state = []
		
		for player in self._players:
			player_state = [player.position, self.encode_orientation(player.orientation), int(player.is_holding_object())]
			for other_player in self._players:
				if other_player.id != player.id:
					player_state.append(other_player.position)
					player_state.append(self.encode_orientation(other_player.orientation))
					player_state.append(int(other_player.is_holding_object()))
			for obj in self._objects:
				if obj.hold_state == HoldState.DISPOSED:  # When disposed, object position is a virtual trash bin at (-1, -1)
					player_state.append((-1, -1))
				else:
					player_state.append(obj.position)
				hold_state = [0] * len(HoldState)
				hold_state[obj.hold_state] = 1
				player_state.append(*hold_state)
				if obj.hold_state == HoldState.FREE:  # If the object is not held, the agent holding it is 0 (None)
					player_state.append([0] * self._n_players)
				else:
					hold_player = [0] * self._n_players
					hold_player[self._players.index(obj.holding_player)] = 1
					player_state.append(*hold_player)
			
			state.append(player_state)
		
		return np.array(state)
	
	def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
		
		if seed is not None:
			self.seed(seed)
		
		self._current_step = 0
		self._n_players = 0
		self._n_objects = 0
		self._players: List[PlayerState] = []
		self._objects: List[ObjectState] = []
		self._field: np.ndarray = np.zeros((self._rows, self._cols))
		self.setup_env()
		
		return self.make_obs(), {}
	
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
		for act_idx in range(self.n_players):
			act = actions[act_idx]
			acting_player = self._players[act_idx]
			act_direction = ActionDirection[Actions(act).name].value
			if act != Actions.INTERACT and act != Actions.STAY:
				acting_player.orientation = act_direction
			next_pos = (max(min(acting_player.position[0] + act_direction[0], self._rows), 0),
						max(min(acting_player.position[1] + act_direction[1], self.cols), 0))
			if not (self._field[next_pos] == CellEntity.EMPTY or self._field[next_pos] == CellEntity.TOXIC or self._field[next_pos] == CellEntity.ICE):
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
						if agent_type == AgentType.ROBOT and (
								agent_action == Actions.STAY or agent_action == Actions.INTERACT):  # check if the agent is a robot and is not trying to move
							if self.require_facing and not self.are_facing(acting_player, agent_facing):
								continue
							# Place object in robot
							place_obj = acting_player.held_objects[0]
							acting_player.drop_object(place_obj.id)
							place_obj.hold_state = HoldState.DISPOSED
							place_obj.holding_player = agent_facing
							place_obj.position = (-1, -1)
							agent_facing.hold_object(place_obj)
							agents_disposed_waste.append(acting_player)
							agents_disposed_waste.append(agent_facing)
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
								acting_player.hold_object(pick_obj)
		
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
			if old_pos != next_pos:
				moving_player.position = next_pos
				if moving_player.is_holding_object():
					for obj in moving_player.held_objects:
						if obj.hold_state != HoldState.DISPOSED:
							obj.position = next_pos
		
		for player in self._players:
			if self.is_game_finished():
				player.reward = ROOM_CLEAN
			elif player in agents_disposed_waste:
				player.reward = DELIVER_WASTE
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
	
	def close_render(self):
		self._render.close()
		
	def close(self):
		super().close()
		self.close_render()