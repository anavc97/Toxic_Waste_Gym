#! /usr/bin/env python
import numpy as np

from src.env.astro_waste_env import Actions, AstroWasteEnv
from typing import List

RNG_SEED = 12072023
N_CYCLES = 100
ACTION_MAP = {'w': Actions.UP, 's': Actions.DOWN, 'a': Actions.LEFT, 'd': Actions.RIGHT, 'q': Actions.STAY, 'e': Actions.INTERACT}


def get_history_entry(obs: np.ndarray, actions: List[int], n_agents: int) -> List:
	entry = []
	for a_idx in range(n_agents):
		state_str = ' '.join([str(int(x)) for x in obs[a_idx][0]])
		action = actions[a_idx]
		entry += [state_str, str(action)]
	
	return entry


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
	encoding = False
	
	env = AstroWasteEnv(field_size, layout, n_players, has_slip, n_objects, max_episode_steps, RNG_SEED, facing, layer_obs, centered_obs, encoding)
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
			action = input('%s action: ' % env.players[idx].name)
			actions += [int(ACTION_MAP[action])]

		print(' '.join([Actions(action).name for action in actions]))
		print(env.objects)
		state, rewards, dones, _, info = env.step(actions)
		print(env.objects)
		print(rewards)
		print(env.objects)
		# print(env.get_filled_field())
		env.render()
		if env.is_over():
			break

	env.close()

if __name__ == '__main__':
	main()
