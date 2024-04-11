import argparse
import json

# Initialize lists to store the extracted information
player_pos_human = []
player_pos_astro = []
object_positions = {}
object_hold_states = {}
object_identified_states = {}
scores = []
timelefts = []

# Create argument parser
parser = argparse.ArgumentParser(description='Parse JSON log file.')
parser.add_argument('logfile', metavar='LOGFILE', type=str, help='Path to JSON log file')

# Parse command line arguments
args = parser.parse_args()

# Open the JSON file
with open(args.logfile, 'r') as file:
    # Iterate through each line in the file
    for line in file:
        # Parse the JSON data in each line
        data = json.loads(line)
        
        # Extract player positions
        player_pos_human.append(data['players'][0]['position'])
        player_pos_astro.append(data['players'][1]['position'])
        
        # Extract object information
        for obj in data['objects']:
            obj_name = obj['name']
            obj_pos = obj['position']
            obj_hold_state = obj['hold_state']
            obj_identified = obj['identified']
            
            # Store object positions
            if obj_name not in object_positions:
                object_positions[obj_name] = []
            object_positions[obj_name].append(obj_pos)
            
            # Store object hold states
            if obj_name not in object_hold_states:
                object_hold_states[obj_name] = []
            object_hold_states[obj_name].append(obj_hold_state)
            
            # Store object identified states
            object_identified_states[obj_name].append(obj_identified)
        
        # Extract score and timeleft
        scores.append(data['score'])
        timelefts.append(data['timeleft'])

# Example usage:
print("Human Player Positions:", player_pos_human)
print("Astro Player Positions:", player_pos_astro)
print("Object Positions:", object_positions)
print("Object Hold States:", object_hold_states)
print("Object Identified States:", object_identified_states)
print("Scores:", scores)
print("Timelefts:", timelefts)
