#! /usr/bin/env python
import numpy as np

from src.env.toxic_waste_env_v2 import Actions, ToxicWasteEnvV2
from typing import List, Tuple
from pathlib import Path

RNG_SEED = 12072023
N_CYCLES = 100
ACTION_MAP = {'w': Actions.UP, 's': Actions.DOWN, 'a': Actions.LEFT, 'd': Actions.RIGHT, 'q': Actions.STAY, 'e': Actions.INTERACT, 'z': Actions.IDENTIFY}
np.set_printoptions(precision=3, linewidth=2000, threshold=1000)


def get_history_entry(obs: np.ndarray, actions: List[int], n_agents: int) -> List:
	entry = []
	for a_idx in range(n_agents):
		state_str = ' '.join([str(int(x)) for x in obs[a_idx][0]])
		action = actions[a_idx]
		entry += [state_str, str(action)]
	
	return entry


def get_model_obs(raw_obs) -> Tuple[np.ndarray, np.ndarray]:
	conv_obs = []
	arr_obs = []

	if isinstance(raw_obs[0], dict):
		conv_obs = raw_obs[0]['conv']
		arr_obs = np.array(raw_obs[0]['array'])
	else:
		conv_obs = raw_obs[0][0].reshape(1, *raw_obs[0].shape)
		arr_obs = raw_obs[0][1:]
	return conv_obs.reshape(1, *conv_obs.shape), arr_obs


def main():
	
	field_size = (15, 15)
	layout = 'cramped_room'
	n_players = 2
	has_slip = False
	n_objects = 4
	max_episode_steps = 500
	facing = True
	layer_obs = True
	centered_obs = False
	use_render = True
	render_mode = ['human']
	data_dir = Path(__file__).parent.absolute().parent.absolute() / 'data'
	
	env = ToxicWasteEnvV2(field_size, layout, n_players, n_objects, max_episode_steps, RNG_SEED, data_dir, facing, centered_obs, render_mode, use_render,
						  slip=has_slip, is_train=True, pick_all=True)
	env.seed(RNG_SEED)
	obs, *_ = env.reset()
	# print(env.get_filled_field())
	env.render()
	
	for i in range(N_CYCLES):

		print('Iteration: %d' % (i + 1))
		# actions = [np.random.choice(range(6)) for _ in range(n_players)]
		actions = []
		print('\n'.join(['Player %s at (%d, %d) with orientation (%d, %d)' % (env.players[idx].name, *env.players[idx].position, *env.players[idx].orientation)
			   for idx in range(n_players)]))
		for idx in range(n_players):
			valid_action = False
			while not valid_action:
				human_input = input("Action for agent %s:\t" % env.players[idx].name)
				try:
					action = int(ACTION_MAP[human_input])
					if action < len(ACTION_MAP):
						valid_action = True
						actions.append(action)
					else:
						print('Action ID must be between 0 and %d, you gave ID %d' % (len(ACTION_MAP), action))
				except KeyError as e:
					print('Key error caught: %s' % str(e))
		print(' '.join([Actions(action).name for action in actions]))
		print(env.objects)
		state, rewards, dones, _, info = env.step(actions)
		next_v2_obs = get_model_obs(state)
		print(next_v2_obs[0].shape)
		print(env.objects)
		print(rewards, dones)
		print(env.objects)
		# print(env.get_filled_field())
		env.render()
		if dones:
			obs, *_ = env.reset()
			env.render()

	env.close()

if __name__ == '__main__':
	main()
