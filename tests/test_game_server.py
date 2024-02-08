#! /usr/bin/env python

import json
import socket
import threading
import time
import sys
import logging

from enum import Enum
from src.env.toxic_waste_env_v1 import AgentType, Actions
from pathlib import Path
from datetime import datetime


RNG_SEED = 12012024
SOCKETS_IP = "127.0.0.1"
INBOUND_PORT = 20501
OUTBOUND_PORT = 20500
SOCK_TIMEOUT = 5
BUFFER_SIZE = 1024
ACTION_MAP = {'w': Actions.UP, 's': Actions.DOWN, 'a': Actions.LEFT, 'd': Actions.RIGHT, 'q': Actions.STAY, 'e': Actions.INTERACT}


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


def process_inbound(sock: socket.socket, stop_condition, logger: logging.Logger):
	try:
		conn, addr = sock.accept()
		print('Receiving updates from front end at %s:%d' % (addr[0], addr[1]))
		logger.info('Receiving updates from front end at %s:%d' % (addr[0], addr[1]))
		with conn:
			while not stop_condition.is_set():
				try:
					message = conn.recv(BUFFER_SIZE).decode('utf-8')
				except socket.timeout as e:
					logger.info("[TIMEOUT] Socket timeout after %d seconds" % SOCK_TIMEOUT)
					message = 'TIMEOUT'
				
				except socket.error as e:
					logger.info("[ERROR] Socket error: %s" % e)
					message = 'ERROR'
				
				logger.info('Got message: %s' % message)
			
			conn.close()
	
	except KeyboardInterrupt:
		print('Exiting test listener')


def main():
	now = datetime.now()
	log_dir = Path(__file__).parent.absolute().parent.absolute() / 'logs'
	log_filename = ('test_game_server' + '_' + now.strftime("%Y%m%d-%H%M%S"))
	logging.basicConfig(filename=(log_dir / (log_filename + '_log.txt')), filemode='w', format='%(asctime)s %(levelname)s:\t%(message)s',
						datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
	logger = logging.getLogger('game_logger')
	
	# Create inbound and outbound TCP sockets
	inbound_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
	outbound_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
	
	# Inbound has to set up a server side connection
	inbound_socket.bind((SOCKETS_IP, INBOUND_PORT))
	inbound_socket.listen()
	print('Inbound socket started at %s:%s' % (SOCKETS_IP, INBOUND_PORT))
	logger.info('Inbound socket started at %s:%s' % (SOCKETS_IP, INBOUND_PORT))
	
	# Outbound only has to connect to front end socket
	outbound_socket.connect((SOCKETS_IP, OUTBOUND_PORT))
	print('Outbound socket connected at: %s:%s' % (SOCKETS_IP, OUTBOUND_PORT))
	logger.info('Outbound socket connected at: %s:%s' % (SOCKETS_IP, OUTBOUND_PORT))
	print("Sockets up and running")
	logger.info("Sockets up and running")
	
	# Create thread for inbound socket to listen for commands
	stop_socket = threading.Event()
	in_thread = threading.Thread(target=process_inbound, args=(inbound_socket, stop_socket, logger))
	in_thread.start()
	
	# Main game cycle
	try:
		outbound_socket.send(json.dumps({'command': str(GameOperations.ADD_PLAYER.value),
											'data': {'name': 'human', 'id': 0, 'type': AgentType.HUMAN}}).encode('utf-8'))
		time.sleep(0.5)
		outbound_socket.send(json.dumps({'command': str(GameOperations.ADD_PLAYER.value),
											'data': {'name': 'robot', 'id': 1, 'type': AgentType.ROBOT}}).encode('utf-8'))
		while True:
			try:
				action = input('Action: ')
				if action.find('human') != -1:
					h_action = action.split(' ')[1]
					outbound_socket.sendall(
						json.dumps({'command': str(GameOperations.PLAYER_ACTION.value), 'data': {'id': 0, 'action': int(ACTION_MAP[h_action])}}).encode('utf-8'))
				elif action.find('robot') != -1:
					r_action = action.split(' ')[1]
					outbound_socket.sendall(
						json.dumps({'command': str(GameOperations.PLAYER_ACTION.value), 'data': {'id': 1, 'action': int(ACTION_MAP[r_action])}}).encode('utf-8'))
				elif action.find('pause') != -1:
					outbound_socket.sendall(json.dumps({'command': str(GameOperations.PAUSE_GAME.value), 'data': ''}).encode('utf-8'))
				elif action.find('unpause') != -1:
					outbound_socket.sendall(json.dumps({'command': str(GameOperations.UNPAUSE_GAME.value), 'data': ''}).encode('utf-8'))
				elif action.find('quit') != -1:
					outbound_socket.sendall(json.dumps({'command': str(GameOperations.CLOSE_GAME.value), 'data': ''}).encode('utf-8'))
				elif action.find('start') != -1:
					outbound_socket.sendall(json.dumps({'command': str(GameOperations.START_GAME.value), 'data': ''}).encode('utf-8'))
			
			except socket.error as e:
				print("[ERROR] Socket error: %s" % e)
				
			except Exception as e:
				print("[ERROR] General error: %s" % e)
			
	except KeyboardInterrupt:
		print('Received Keyboard Interrupt, closing game and sockets')
		inbound_socket.close()
		outbound_socket.close()
		stop_socket.set()
		sys.exit(0)


if __name__ == '__main__':
	main()