#! /usr/bin/env python

import pyglet
import json

from pathlib import Path
from pyglet.gl import *


def main():

	icons_dir = Path(__file__).parent.absolute() / 'env' / 'data' / 'icons'
	with open(icons_dir / 'sprites_info.json', 'r') as a_info_file:
		agents_info = json.load(a_info_file)['frames']
	with open(icons_dir / 'terrain.json', 'r') as a_info_file:
		terrain_info = json.load(a_info_file)['frames']
	with open(icons_dir / 'objects.json', 'r') as a_info_file:
		ball_info = json.load(a_info_file)['frames']
	
	pyglet.resource.path = [str(icons_dir)]
	pyglet.resource.reindex()
	# img_astro = pyglet.resource.image("astros.png")
	# img_human = pyglet.resource.image("humans.png")
	img_balls = pyglet.resource.image("objects.png")
	img_terrain = pyglet.resource.image("terrain.png")
	
	# print(img_astro)
	# for key in agents_info.keys():
	# 	sprite_info = agents_info[key]['frame']
	# 	sprite_pos = (sprite_info['x'], sprite_info['y'])
	# 	sprite_dims = (sprite_info['w'], sprite_info['h'])
	# 	print(key, sprite_pos, sprite_dims)
	# 	astro_sprite = img_astro.get_region(sprite_info['x'], img_astro.height - sprite_info['y'] - 14, sprite_info['w'], sprite_info['h'])
	# 	astro_sprite.save(icons_dir / 'crop' / ('astro-%s' % key))
	# 	human_sprite = img_human.get_region(sprite_info['x'], img_human.height - sprite_info['y'] - 14, sprite_info['w'], sprite_info['h'])
	# 	human_sprite.save(icons_dir / 'crop' / ('human-%s' % key))

	print(img_balls)
	for key in terrain_info.keys():
		sprite_info = terrain_info[key]['frame']
		sprite_pos = (sprite_info['x'], sprite_info['y'])
		sprite_dims = (sprite_info['w'], sprite_info['h'])
		print(key, sprite_pos, sprite_dims)
		astro_sprite = img_terrain.get_region(sprite_info['x'], img_terrain.height - sprite_info['y'] - 14, sprite_info['w'], sprite_info['h'])
		astro_sprite.save(icons_dir / 'crop' / ('%s' % key))
	
	print(img_terrain)
	for key in ball_info.keys():
		sprite_info = ball_info[key]['frame']
		sprite_pos = (sprite_info['x'], sprite_info['y'])
		sprite_dims = (sprite_info['w'], sprite_info['h'])
		print(key, sprite_pos, sprite_dims)
		astro_sprite = img_balls.get_region(sprite_info['x'], img_balls.height - sprite_info['y'] - 14, sprite_info['w'], sprite_info['h'])
		astro_sprite.save(icons_dir / 'crop' / ('%s' % key))

if __name__ == '__main__':
	main()