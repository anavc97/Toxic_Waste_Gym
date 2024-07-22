#! /usr/bin/env python

import numpy as np

from src.env.toxic_waste_env_base import PlayerState
from src.env.toxic_waste_env_v2 import WasteStateV2, ToxicWasteEnvV2, Actions
from src.env.astro_greedy_agent import GreedyAgent
from itertools import permutations
from pathlib import Path


RNG_SEED = 18102023
N_CYCLES = 250
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
	use_render = True
	rng_gen = np.random.default_rng(RNG_SEED)
	agent_models = []
	data_dir = Path(__file__).parent.absolute().parent.absolute() / 'data'
	env = ToxicWasteEnvV2(field_size, layout, n_players, n_objects, max_episode_steps, RNG_SEED, data_dir, centered_obs,
	                      slip=has_slip, is_train=True, use_render=use_render, pick_all=True)

	state, *_ = env.reset(seed=RNG_SEED)
	waste_idx = []
	for obj in env.objects:
		waste_idx.append(env.objects.index(obj))
	waste_seqs = list(permutations(waste_idx))
	# waste_order = list(np.random.default_rng().choice(np.array(waste_seqs)))
	waste_order = list(rng_gen.choice(np.array(waste_seqs)))
	for player in env.players:
		agent_models.append(GreedyAgent(player.position, player.orientation, player.name,
										dict([(idx, env.objects[idx].position) for idx in range(len(env.objects))]), RNG_SEED, env.field, 2,
										env.door_pos, agent_type=player.agent_type))
	for model in agent_models:
		model.waste_order = waste_order
	# print(env.get_filled_field())
	env.render()
	successes = 0
	
	for i in range(N_CYCLES):

		done = False
		print('Iteration: %d' % (i + 1))
		epoch = 0
		print(agent_models[0].waste_order)
		while not done:
			print('Epoch %d' % (epoch + 1))
			actions = []
			for idx in range(n_players):
				# action = rng_gen.choice(len(Actions))
				action = agent_models[idx].act(env.create_observation())
				print('Player %s at (%d, %d) with orientation (%d, %d) and chose action %s with plan %s' % (env.players[idx].name, *env.players[idx].position,
																											*env.players[idx].orientation, Actions(action).name,
																											agent_models[idx].plan))
				print('Wastes: ', str(env.objects))
				actions += [action]
	
			state, rewards, finished, timeout, info = env.step(actions)
			env.render()
			epoch += 1
			# print(env.get_filled_field())
			if finished or timeout:
				done = True
				env.reset()
				[model.reset(waste_order, dict([(idx, env.objects[idx].position) for idx in range(env.n_objects)]), env.has_pick_all) for model in agent_models]
				if finished:
					successes += 1
			input()

	print('Finished %f of attempts' % (successes / N_CYCLES))
	



if __name__ == '__main__':
	main()
