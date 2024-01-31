import UnityEngine as ue
import json
from create_sockets import GameOperations, outbound_socket
'''
data = {"command":"start_game", "data": {}}

file_name = "start_game.json"

with open(file_name, 'w') as json_file:
    json.dump(data, json_file)
'''
outbound_socket.sendall(json.dumps({'command': str(GameOperations.START_GAME.value), 'data': ''}).encode('utf-8'))
ue.Debug.Log("Game Started on Unity side")