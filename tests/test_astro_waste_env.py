#! /usr/bin/env python

import numpy as np
import yaml

from env.astro_waste_env import AstroWasteEnv, PlayerState, ObjectState, Actions
from pathlib import Path


RNG_SEED = 18102023
N_CYCLES = 10
ACTION_MAP = {'w': Actions.UP, 's': Actions.DOWN, 'a': Actions.LEFT, 'd': Actions.RIGHT, 'q': Actions.STAY}


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
	rng_gen = np.random.default_rng(RNG_SEED)
	env = AstroWasteEnv(field_size, layout, n_players, has_slip, n_objects, max_episode_steps, RNG_SEED, facing, layer_obs, centered_obs, encoding)
	state, *_ = env.reset()
	print(env.get_filled_field())
	
	for i in range(N_CYCLES):

		print('Iteration: %d' % (i + 1))
		actions = []
		for idx in range(n_players):
			action = rng_gen.choice(len(Actions))
			print('Player %s at (%d, %d) with orientation (%d, %d) and chose action %s' % (env.players[idx].name, *env.players[idx].position,
																						   *env.players[idx].orientation, Actions(action).name))
			actions += [action]

		state, rewards, dones, _, info = env.step(actions)
		print(env.get_filled_field())
		for layer in state[1, :]:
			print(layer)
			print('\n')
	



if __name__ == '__main__':
	main()
