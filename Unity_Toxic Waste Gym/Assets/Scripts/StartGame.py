import UnityEngine as ue
import json

data = {"command":"start_game", "data": {}}

file_name = "start_game.json"

with open(file_name, 'w') as json_file:
    json.dump(data, json_file)