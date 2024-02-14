#! /usr/bin/env python

import argparse
import json
import logging
import socket
import threading
import time
import sys

from src.env.toxic_waste_env_v1 import ToxicWasteEnvV1, Actions
from src.env.astro_waste_game import AstroWasteGame
from enum import Enum
from datetime import datetime
from pathlib import Path


RNG_SEED = 12012024
SOCKETS_IP = "127.0.0.1"
INBOUND_PORT = 20500
OUTBOUND_PORT = 20501
SOCK_TIMEOUT = 5
BUFFER_SIZE = 1024


class GameOperations(Enum):
	
	ADD_PLAYER = 'n_play'
	REMOVE_PLAYER = 'r_play'
	PLAYER_ACTION = 'p_act'
	PAUSE_GAME = 'pause'
	UNPAUSE_GAME = 'unpause'
	CLOSE_GAME = 'close'
	RESTART_LEVEL = 'reset_lvl'
	RESTART_GAME = 'reset_game'
	START_GAME = 'start_game'


def process_inbound(sock: socket.socket, stop_condition, game: AstroWasteGame, logger: logging.Logger, close_game: bool):
	conn, addr = sock.accept()
	logger.info('Receiving updates from front end at %s:%d' % (addr[0], addr[1]))
	with conn:
		while not stop_condition.is_set():
			try:
				# conn.settimeout(SOCK_TIMEOUT)
				message = conn.recv(BUFFER_SIZE).decode('utf-8')
			
			except socket.timeout as e:
				logger.error("[INBOUND TIMEOUT] Socket timeout after %d seconds" % SOCK_TIMEOUT)
				message = 'TIMEOUT'
			
			except socket.error as e:
				logger.error("[INBOUND ERROR] Socket error: %s" % e)
				message = 'ERROR'
				
			except Exception as e:
				logger.error("[INBOUND ERROR] General error: %s" % e)
			
			if message == 'TIMEOUT' or message == 'ERROR' or message == '':	# On error or timeout do nothing
				pass
			
			else:															# If a message was received process to find the command sent
				json_message = json.loads(message)
				command = json_message['command']
				data = json_message['data']
				logger.info('Received command: %s and data %s' % (command, str(data)))
				print('Received command: %s and data %s' % (command, str(data)))
				
				if command == GameOperations.ADD_PLAYER.value:
					p_id = int(data['id'])
					p_name = data['name']
					p_type = int(data['type'])
					response = game.add_player(p_id, p_name, p_type)
					logger.info(response)
				elif command == GameOperations.REMOVE_PLAYER.value:
					p_id = int(data['id'])
					response = game.remove_player(p_id)
					logger.info(response)
				elif command == GameOperations.PLAYER_ACTION.value:
					p_id = int(data['id'])
					action = int(data['action'])
					response = game.enque_action(p_id, action)
					logger.info(response)
				elif command == GameOperations.PAUSE_GAME.value:
					response = game.pause_game()
					logger.info(response)
				elif command == GameOperations.UNPAUSE_GAME.value:
					response = game.unpause_game()
					logger.info(response)
				elif command == GameOperations.RESTART_GAME.value:
					game.level_idx = 0
					game.env_reset()
				elif command == GameOperations.RESTART_LEVEL.value:
					game.env_reset()
				elif command == GameOperations.START_GAME.value:
					game.start_game()
				elif command == GameOperations.CLOSE_GAME.value:
					close_game = True
				else:
					logger.info('Received unknown command %s, ignoring...' % command)
					
		conn.close()


def main():
	parser = argparse.ArgumentParser(description='Script to test the game server backend running locally to the front-end')
	
	parser.add_argument('--field-size', dest='field_size', type=int, required=True, nargs=2, help='Number of rows and cols in the environment')
	parser.add_argument('--max-env-players', dest='max_env_players', type=int, required=True,
						help='Maximum number of players in the environment at any given time')
	parser.add_argument('--max-game-players', dest='max_game_players', type=int, required=True,
						help='Maximum number of players in the game at any given time')
	parser.add_argument('--max-objects', dest='max_objects', type=int, required=True,
						help='Maximum number of objects in the game at any given time')
	parser.add_argument('--max-steps', dest='max_steps', type=int, required=True, help='Maximum number of steps before environment times out')
	parser.add_argument('--cycles-second', dest='cycles_second', type=int, required=True, help='Number of game cycles per second')
	parser.add_argument('--game-id', dest='game_id', type=int, required=False, default=0, help='Integer identifying current game')
	parser.add_argument('--levels', dest='game_levels', type=str, required=True, nargs='+', help='List of levels for the game')
	parser.add_argument('--slip', dest='has_slip', action='store_true', help='Flag that triggers slippage in the environment')
	parser.add_argument('--require-facing', dest='require_facing', action='store_true',
						help='Flag that forces agents to face each other to deposit balls')
	parser.add_argument('--layer-obs', dest='use_layers', action='store_true', help='Environment observation in layer organization')
	parser.add_argument('--centered-obs', dest='centered_obs', action='store_true', help='Environment observations are centered on each agent')
	parser.add_argument('--use-encoding', dest='use_encoding', action='store_true', help='Use one-hot encoding for categorical observations')
	parser.add_argument('--render-mode', dest='render_mode', type=str, nargs='+', required=False, default=None,
						help='List of render modes for the environment')
	args = parser.parse_args()
	
	env = ToxicWasteEnvV1(args.field_size, args.game_levels[0], args.max_env_players, args.max_objects, args.max_steps, RNG_SEED, args.require_facing,
						  args.use_layers, args.centered_obs, args.use_encoding, args.render_mode, slip=args.has_slip)
	game = AstroWasteGame(args.cycles_second, args.game_levels, env, args.max_game_players, args.game_id)
	if args.render_mode and 'human' in args.render_mode:
		render = True
	else:
		render = False
	close_game = False
	now = datetime.now()
	log_dir = Path(__file__).parent.absolute().parent.absolute() / 'logs'
	log_filename = ('game-%d' % args.game_id + '_' + now.strftime("%Y%m%d-%H%M%S"))
	if len(logging.root.handlers) > 0:
		for handler in logging.root.handlers:
			logging.root.removeHandler(handler)
	
	logging.basicConfig(filename=(log_dir / (log_filename + '_log.txt')), filemode='w', format='%(asctime)s %(levelname)s:\t%(message)s',
						datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
	logger = logging.getLogger('game_logger')
	
	# Create inbound and outbound TCP sockets
	inbound_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
	outbound_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
	initialized_outbound = False
	
	# Inbound has to set up a server side connection
	inbound_socket.bind((SOCKETS_IP, INBOUND_PORT))
	inbound_socket.listen()
	logger.info('Inbound socket started at %s:%s' % (SOCKETS_IP, INBOUND_PORT))
	
	
	# Create thread for inbound socket to listen for commands
	stop_socket = threading.Event()
	in_thread = threading.Thread(target=process_inbound, args=(inbound_socket, stop_socket, game, logger, close_game))
	in_thread.start()
	i = 0
	# Main game cycle
	try:
		obs, *_ = game.env_reset()
		while not (game.game_finished() or close_game):
			if not game.game_started:	# Only works when game has started
				continue

			else:
				if render:
					env.render()
				if not initialized_outbound:
					# Outbound only has to connect to front end socket
					print("Waiting a bit for Unity.")
					time.sleep(1)
					print("Ready to connect.")
					outbound_socket.connect((SOCKETS_IP, OUTBOUND_PORT))
					logger.info('Outbound socket connected at: %s:%s' % (SOCKETS_IP, OUTBOUND_PORT))
					initialized_outbound = True

				# After waking up get robot action and run environment step
				# TODO: Get robot action
				obs, _, actions = game.env_step()
				try:
					# When game finishes, send message to front end warning that the game is over
					if game.game_finished():
						logger.info("game finished")
						out_msg = json.dumps({'command': 'game_finished', 'data': ''})

					# When level has finished, change to next level, reset environment and send message with new level to front end
					elif game.level_finished():
						logger.info("level finished")
						game.level_idx += 1
						nxt_level = game.levels[game.level_idx]
						game.game_env.layout = nxt_level
						game.env_reset()
						out_msg = json.dumps({'command': 'new_level', 'data': nxt_level})

					# Standard behaviour sends message with new state data to front end
					else:
						new_state = game.get_game_metadata()
						out_msg = json.dumps({'command': 'new_state', 'data': new_state})
						logger.info("new state: ", new_state)

					i += 1
					out_msg = out_msg + "<EOF>"
					logger.info(out_msg)
					outbound_socket.sendall(out_msg.encode('utf-8'))

				except socket.error as e:
					logger.error("[MAIN ERROR] Socket error: %s" % e)
			
			# TODO: log game data for step
			time.sleep(1 / float(args.cycles_second))

		logger.info('Game is over, closing sockets and threads')
		inbound_socket.close()
		outbound_socket.close()
		stop_socket.set()
		sys.exit(0)

	except KeyboardInterrupt:
		logger.info('Received Keyboard Interrupt, closing game and sockets')
		inbound_socket.close()
		outbound_socket.close()
		stop_socket.set()
		sys.exit(0)


if __name__ == '__main__':
	main()