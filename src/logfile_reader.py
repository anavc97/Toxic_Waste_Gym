import argparse
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from natsort import natsorted
import matplotlib.patches as patches

LAYOUTS = ["level_zero", "level_one", "level_two", "level_three"]

# Initialize lists to store the extracted information
trajectories_good = {"level_zero":[],"level_one":[],"level_two":[],"level_three":[] }
mean_distance_to_astro_good = {}
identified_dep_balls_good = {}
unidentified_dep_balls_good = {}
undep_balls_good = {}
total_time_good = {}
wall_positions ={"level_zero":[],"level_one":[],"level_two":[[],[],[]],"level_three":[] }

# List of wall coordinates for level two
walls_level_two = [
    ([4, 3], [5, 3]),
    ([9, 3], [10, 3]),
    ([2, 4], [7, 4]),
    ([2, 5], [5, 5]),
    ([9, 5], [11, 5]),
    ([2, 6], [5, 6]),
    ([7, 6], [11, 6]),
    ([2, 7], [3, 7]),
    ([5, 9], [11, 9]),
    ([1, 10], [3, 10]),
    ([5, 10], [11, 10]),
    ([2, 12], [4, 12])
]

# Function to generate wall positions from start and end coordinates
def generate_wall_positions(start, end):
    positions = []
    if start[0] == end[0]:  # Vertical wall
        for y in range(start[1], end[1] + 1):
            positions.append((start[0], y))
    elif start[1] == end[1]:  # Horizontal wall
        for x in range(start[0], end[0] + 1):
            positions.append((x, start[1]))
    return positions

# Populate wall_positions for level two
for wall in walls_level_two:
    start, end = wall
    wall_positions["level_two"].extend(generate_wall_positions(start, end))


trajectories_bad = {"level_zero":[],"level_one":[],"level_two":[],"level_three":[] }
mean_distance_to_astro_bad = {}
identified_dep_balls_bad = {}
unidentified_dep_balls_bad = {}
undep_balls_bad = {}
total_time_bad = {}

player_ids_good = []
player_ids_bad = []

player_pos_human = []
timesteps = []
player_pos_astro = []
player_orientation_human = []
player_orientation_astro = []
object_positions = {

}
object_hold_states = {}
object_identified_states = {}
scores = []
timelefts = []
dep_balls = []
time_spent_player_good = {}
positions_holding_ball_good = {}
last_timestamp = None
last_color = None
player_count = 0

# Create argument parser
parser = argparse.ArgumentParser(description='Parse JSON log files in a directory.')
parser.add_argument('--logdir', dest='logdir', type=str, required=True, help='Path to directory containing JSON log files')
parser.add_argument('--plot', dest='plot', action='store_true', help='If plot trajectories or not')

# Parse command line arguments
args = parser.parse_args()
logfile_dir = args.logdir
dir_good = logfile_dir + "/Good"
dir_bad = logfile_dir + "/Bad"

def sort_entries_by_time(entries):
    return sorted(entries, key=lambda x: datetime.strptime(x["time"], "%Y-%m-%d %H:%M:%S.%f"))

def convert_position(pos):
    return [14 - pos[1], pos[0]]

def convert_orientation(orientation):
    return [-orientation[1], orientation[0]]

def convert_to_env_grid(entries):
    converted_entries = []
    # Convert players
    for entry in entries:
        for player in entry['players']:
            player['position'] = convert_position(player['position'])
            player['orientation'] = convert_orientation(player['orientation'])
        
        # Convert objects
        for obj in entry['objects']:
            obj['position'] = convert_position(obj['position'])

        converted_entries.append(entry)
    return converted_entries

def write_sorted_logfile(entries, output_path):
    with open(output_path, 'w') as file:
        for entry in entries:
            file.write(json.dumps(entry) + '\n')

def read_logfile(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [json.loads(line) for line in lines]

def sort_logfile(input_path, output_path):
    data = read_logfile(input_path)
    sorted_entries = sort_entries_by_time(data)
    converted_entries = convert_to_env_grid(sorted_entries)
    write_sorted_logfile(converted_entries, output_path)

def process_mean_dist_to_astro(level, player_trajectory, astro_trajectory, cond, id):
        # Ensure both trajectories have the same length
    min_length = min(len(player_trajectory), len(astro_trajectory))
    player_trajectory = np.array(player_trajectory[:min_length])
    astro_trajectory = np.array(astro_trajectory[:min_length])

    # Calculate distances
    distances = np.linalg.norm(player_trajectory - astro_trajectory, axis=1)
    
    # Calculate mean distance
    mean_distance = np.mean(distances)
    
    if cond == "Good":
        # Append the mean distance to the corresponding level in the dictionary
        mean_distance_to_astro_good[id][level].append(mean_distance)
    elif cond == "Bad":
        mean_distance_to_astro_bad[id][level].append(mean_distance)

def count_colors_by_level(data):
    colors = ['red', 'yellow', 'green']
    for level in data:
        color_counts = {color: 0 for color in colors}
        for lst in data[level]:
            for item in lst:
                for color in colors:
                    if item.startswith(color):
                        color_counts[color] += 1
        print(f"Counts in {level}: {color_counts}")

def count_balls(ball_list):
    counts = {'red': 0, 'green': 0, 'yellow': 0}
    for ball in ball_list[0]:
        if 'red' in ball:
            counts['red'] += 1
        elif 'green' in ball:
            counts['green'] += 1
        elif 'yellow' in ball:
            counts['yellow'] += 1
    return counts

def print_ball_counts(COND):
    
    if COND == "Good":
        for player_id, levels in identified_dep_balls_good.items():
            output = [f"{player_id}:"]
            for level in levels.keys():
                id_dep_counts = count_balls(identified_dep_balls_good[player_id][level])
                unid_dep_counts = count_balls(unidentified_dep_balls_good[player_id][level])
                undep_counts = count_balls(undep_balls_good[player_id][level])
                output.append(f"{level}_id_red: {id_dep_counts['red']}: {level}_id_green: {id_dep_counts['green']}: {level}_id_yellow: {id_dep_counts['yellow']}:")
                output.append(f"{level}_unid_red: {unid_dep_counts['red']}: {level}_unid_green: {unid_dep_counts['green']}: {level}_unid_yellow: {unid_dep_counts['yellow']}:")
                output.append(f"{level}_undep_red: {undep_counts['red']}: {level}_undep_green: {undep_counts['green']}: {level}_undep_yellow: {undep_counts['yellow']}")
            print(" ".join(output))     
    
    elif COND == "Bad":
        for player_id, levels in identified_dep_balls_bad.items():
            output = [f"{player_id}:"]
            for level in levels.keys():
                id_dep_counts = count_balls(identified_dep_balls_bad[player_id][level])
                unid_dep_counts = count_balls(unidentified_dep_balls_bad[player_id][level])
                undep_counts = count_balls(undep_balls_bad[player_id][level])
                output.append(f"{level}_id_red: {id_dep_counts['red']}: {level}_id_green: {id_dep_counts['green']}: {level}_id_yellow: {id_dep_counts['yellow']}:")
                output.append(f"{level}_unid_red: {unid_dep_counts['red']}: {level}_unid_green: {unid_dep_counts['green']}: {level}_unid_yellow: {unid_dep_counts['yellow']}:")
                output.append(f"{level}_undep_red: {undep_counts['red']}: {level}_undep_green: {undep_counts['green']}: {level}_undep_yellow: {undep_counts['yellow']}")
            
            print(" ".join(output))
   
def parse_time_balls(log_entry):
    entry = json.loads(log_entry)
    players = entry["players"]
    objects = entry["objects"]
    timestamp = datetime.strptime(entry["time"], "%Y-%m-%d %H:%M:%S.%f")
    
    human_held_object = None
    for player in players:
        if player["name"] == "human":
            human_held_object = player["held_object"]
            break
    
    object_color = None
    if human_held_object:
        for obj in objects:
            if obj["name"] == human_held_object:
                if "yellow" in obj["name"]:
                    object_color = "yellow"
                elif "green" in obj["name"]:
                    object_color = "green"
                elif "red" in obj["name"]:
                    object_color = "red"
                break
    
    return timestamp, object_color

def calculate_game_duration(start_time, end_time):
    # Define the datetime format
    datetime_format = "%Y-%m-%d %H:%M:%S.%f"
    
    # Parse the start and end times into datetime objects
    start = datetime.strptime(start_time, datetime_format)
    end = datetime.strptime(end_time, datetime_format)
    
    # Calculate the duration
    duration = end - start
    
    # Convert the duration to total minutes as a float
    duration = duration.total_seconds()
    
    return duration

#### PROCESSING GOOD CONDITION

COND = "Good"

# Ensure the directory path ends with a '/'
if not dir_good.endswith('/'):
    dir_good += '/'

files = os.listdir(dir_good)
files.sort()
print("files good: ", files)
input()
# Iterate over all files in the directory
for logfile in files:
    input_filename = os.path.join(dir_good, logfile)
    output_logfile_dir = os.path.join(dir_good, 'corrected')
    output_logfile_path = os.path.join(output_logfile_dir, f'corrected_{logfile}')
    
    # Check if the file is a regular file (not a directory)
    if os.path.isfile(input_filename):
        print("filenames: " + input_filename + " || " + output_logfile_path)
        sort_logfile(input_filename, output_logfile_path)
        player_count += 1
        a = 0
        player_pos_human = []
        timesteps = []
        player_pos_astro = []
        player_orientation_human = []
        player_orientation_astro = []
        object_positions = {

        }
        object_hold_states = {}
        object_identified_states = {}
        scores = []
        timelefts = []
        dep_balls = []

        # PROCESS SORTED DATA
        with open(output_logfile_path, 'r') as file:
            # Iterate through each line in the file
            for line in file:
                data = json.loads(line)
                if a<len(LAYOUTS): l = LAYOUTS[a]
                else: l = LAYOUTS[-1]
                
                if data['layout'] != l: 
                    if args.plot and data['layout'] == "level_three":
                        print("Plotting now...")

                        # Create a figure and axis
                        fig, ax = plt.subplots(figsize=(15, 15))

                        def update(frame):
                            ax.clear()
                            
                            # Plot human positions in brown
                            human_pos = player_pos_human[frame]
                            human_orientation = player_orientation_human[frame]
                            ax.scatter(human_pos[0], human_pos[1], c='brown', label='Human')
                            ax.quiver(human_pos[0], human_pos[1], human_orientation[0], human_orientation[1], angles='xy', scale_units='xy', scale=0.7, color='brown')
                            
                            # Plot astro positions in blu
                            astro_pos = player_pos_astro[frame]
                            astro_orientation = player_orientation_astro[frame]
                            ax.scatter(astro_pos[0], astro_pos[1], c='blue', label='Astro')
                            ax.quiver(astro_pos[0], astro_pos[1], astro_orientation[0], astro_orientation[1], angles='xy', scale_units='xy', scale=0.7, color='blue')
                            
                            # Plot object positions by color
                            for color, positions in object_positions.items():
                                color_map = {'green_1': 'green','green_2': 'green','green_3': 'green', 'yellow_1': 'orange', 'yellow_2': 'orange', 'yellow_3': 'orange', 'red_1': 'red', 'red_2': 'red', 'red_3': 'red'}
                                ax.scatter(positions[frame][0], positions[frame][1], c=color_map[color], label=f'{color.capitalize()} Objects')
                            
                            '''
                            # Add black squares for wall positions
                            for pos in wall_positions[l]:
                                print(pos)
                                input()
                                square = patches.Rectangle((pos[0] - 0.25, pos[1] - 0.25), 0.5, 0.5, linewidth=1, edgecolor='black', facecolor='black')
                                ax.add_patch(square)'''
                            # Set grid limits and labels
                            ax.set_xlim(0, 15)
                            ax.set_ylim(0, 15)
                            ax.set_xticks(range(16))
                            ax.set_yticks(range(16))
                            ax.set_xlabel('X Position')
                            ax.set_ylabel('Y Position')
                            id = data['id']
                            ax.set_title(f'Player {id} Cond:{COND} - {l} (Time: {timesteps[frame]})')
                            ax.grid(True)
                            
                            # Avoid duplicate legend entries
                            handles, labels = plt.gca().get_legend_handles_labels()
                            by_label = dict(zip(labels, handles))
                            ax.legend(by_label.values(), by_label.keys())

                        # Create animation
                        ani = animation.FuncAnimation(fig, update, frames=len(timesteps), repeat=False)

                        # Show the plot
                        plt.show()
                        input("cleck")
                    
                    trajectories_good[l].append(player_pos_human)
                    process_mean_dist_to_astro(l,player_pos_human,player_pos_astro, COND, data['id'])
                    total_time_good[data['id']][l].append(calculate_game_duration(timesteps[0], timesteps[-1]))
                    undep_list = []
                    id_list = []
                    unid_list = []
                    for obj in object_positions.keys():
                        if obj in dep_balls:
                            if True in object_identified_states[obj]:id_list.append(obj)
                            else: unid_list.append(obj)
                        else: undep_list.append(obj)
                    
                    identified_dep_balls_good[data['id']][l].append(id_list)
                    unidentified_dep_balls_good[data['id']][l].append(unid_list)
                    undep_balls_good[data['id']][l].append(undep_list)

                    player_pos_human = []
                    timesteps = []
                    player_pos_astro = []
                    player_orientation_human = []
                    player_orientation_astro = []
                    object_positions = {

                    }
                    object_hold_states = {}
                    object_identified_states = {}
                    scores = []
                    timelefts = []
                    a+=1
                    dep_balls = []
                    
                if data['id'] not in player_ids_good:
                    player_ids_good.append(data['id'])
                    time_spent_player_good[data['id']] = {
                    "level_zero": {"red": 0, "yellow": 0, "green": 0},
                    "level_one": {"red": 0, "yellow": 0, "green": 0},
                    "level_two": {"red": 0, "yellow": 0, "green": 0},
                    "level_three": {"red": 0, "yellow": 0, "green": 0}
                    }
                    positions_holding_ball_good[data['id']] = {
                    "level_zero": [],
                    "level_one": [],
                    "level_two": [],
                    "level_three": []
                    }
                    total_time_good[data['id']] = {
                    "level_zero": [],
                    "level_one": [],
                    "level_two": [],
                    "level_three": []
                    }
                    mean_distance_to_astro_good[data['id']] = {
                    "level_zero": [],
                    "level_one": [],
                    "level_two": [],
                    "level_three": []
                    }
                    identified_dep_balls_good[data['id']] = {
                    "level_zero": [],
                    "level_one": [],
                    "level_two": [],
                    "level_three": []
                    }
                    unidentified_dep_balls_good[data['id']] = {
                    "level_zero": [],
                    "level_one": [],
                    "level_two": [],
                    "level_three": []
                    }
                    undep_balls_good[data['id']] = {
                    "level_zero": [],
                    "level_one": [],
                    "level_two": [],
                    "level_three": []
                    }
                    print(data['id'])

                # Extract player positions
                player_pos_human.append(data['players'][0]['position'])
                player_pos_astro.append(data['players'][1]['position'])
                player_orientation_human.append(data['players'][0]['orientation'])
                player_orientation_astro.append(data['players'][1]['orientation'])
                
                # Extract object information
                for obj in data['objects']:
                    obj_name = obj['name']
                    obj_pos = obj['position']
                    obj_hold_state = obj['hold_state']
                    obj_identified = obj['identified']
                    
                    # Store object positions based on their color
                    if obj_name not in object_positions.keys():
                        object_positions[obj_name] = []
                    object_positions[obj_name].append(obj_pos)
                    
                    # Store object hold states
                    if obj_name not in object_hold_states:
                        object_hold_states[obj_name] = []
                    object_hold_states[obj_name].append(obj_hold_state)
                    
                    # Store object identified states
                    if obj_name not in object_identified_states:
                        object_identified_states[obj_name] = []
                    object_identified_states[obj_name].append(obj_identified)
                
                # Extract score and timeleft
                scores.append(data['score'])
                timelefts.append(data['timeleft'])
                timesteps.append(data['time'])
                if data['players'][1]['held_object'] != None: dep_balls = data['players'][1]['held_object']
                else: dep_balls=[]
                
                #extract time holding ball
                timestamp, color = parse_time_balls(line)
                if data['players'][0]['held_object']: 
                    if data['players'][0]['position'] not in positions_holding_ball_good[data['id']][data['layout']]:
                        positions_holding_ball_good[data['id']][data['layout']].append(data['players'][0]['position'])
    
                if last_timestamp and last_color:
                    time_diff = (timestamp - last_timestamp).total_seconds()
                    time_spent_player_good[data['id']][data['layout']][last_color] += time_diff

    
                last_timestamp = timestamp
                last_color = color
        
        trajectories_good[l].append(player_pos_human)
        total_time_good[data['id']][l].append(calculate_game_duration(timesteps[0], timesteps[-1]))
        process_mean_dist_to_astro(l,player_pos_human,player_pos_astro, COND, data['id'])
        print(len(mean_distance_to_astro_good), len(mean_distance_to_astro_good), len(mean_distance_to_astro_good), len(mean_distance_to_astro_good))
        undep_list = []
        id_list = []
        unid_list = []
        for obj in object_positions.keys():
            if obj in dep_balls:
                if True in object_identified_states[obj]:id_list.append(obj)
                else: unid_list.append(obj)
            else: undep_list.append(obj)
        
        identified_dep_balls_good[data['id']][l].append(id_list)
        unidentified_dep_balls_good[data['id']][l].append(unid_list)
        undep_balls_good[data['id']][l].append(undep_list)

        
        #print("id balls: ", identified_dep_balls_good)
        #print("unid balls: ", unidentified_dep_balls_good)
        #print("undep balls: ", undep_balls_good)
        #print("time holding ball: ", time_spent_good)
        
    
    if args.plot:
        print("Plotting now...")

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(15, 15))

        def update(frame):
            ax.clear()
            
            # Plot human positions in brown
            human_pos = player_pos_human[frame]
            human_orientation = player_orientation_human[frame]
            ax.scatter(human_pos[0], human_pos[1], c='brown', label='Human')
            ax.quiver(human_pos[0], human_pos[1], human_orientation[0], human_orientation[1], angles='xy', scale_units='xy', scale=0.7, color='brown')
            
            # Plot astro positions in blu
            astro_pos = player_pos_astro[frame]
            astro_orientation = player_orientation_astro[frame]
            ax.scatter(astro_pos[0], astro_pos[1], c='blue', label='Astro')
            ax.quiver(astro_pos[0], astro_pos[1], astro_orientation[0], astro_orientation[1], angles='xy', scale_units='xy', scale=0.7, color='blue')
            
            # Plot object positions by color
            for color, positions in object_positions.items():
                color_map = {'green_1': 'green','green_2': 'green','green_3': 'green', 'yellow_1': 'orange', 'yellow_2': 'orange', 'yellow_3': 'orange', 'red_1': 'red', 'red_2': 'red', 'red_3': 'red'}
                ax.scatter(positions[frame][0], positions[frame][1], c=color_map[color], label=f'{color.capitalize()} Objects')
            
            # Set grid limits and labels
            ax.set_xlim(0, 15)
            ax.set_ylim(0, 15)
            ax.set_xticks(range(16))
            ax.set_yticks(range(16))
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_title(f'Player {id} Cond:{COND} - {l} (Time: {timesteps[frame]})')
            ax.grid(True)
            
            # Avoid duplicate legend entries
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(timesteps), repeat=False)

        # Show the plot
        plt.show()

for player_id, levels in mean_distance_to_astro_good.items():
    output = [f"{player_id}:"]
    for level, mean in levels.items():
        output.append(f"mean_dist_{level}: {mean[0]:.2f}:")
    print(" ".join(output))

print("player ids good: ", player_ids_good)
for player_id, levels in time_spent_player_good.items():
    output = [f"{player_id}:"]
    for level, colors in levels.items():
        for color, time in colors.items():
            output.append(f"{level}_{color}: {time}:")
    print(" ".join(output))
########################################################################################################################
#### PROCESSING BAD CONDITION

COND = "Bad"
player_pos_human = []
timesteps = []
player_pos_astro = []
player_orientation_human = []
player_orientation_astro = []
object_positions = {

}
object_hold_states = {}
object_identified_states = {}
scores = []
timelefts = []
dep_balls = []
time_spent_player_bad = {}
positions_holding_ball_bad = {}
last_timestamp = None
last_color = None
# Ensure the directory path ends with a '/'
if not dir_bad.endswith('/'):
    dir_bad += '/'


files = os.listdir(dir_bad)
files.sort()
print("files bad: ", files)
# Iterate over all files in the directory
for logfile in files:
    input_filename = os.path.join(dir_bad, logfile)
    output_logfile_dir = os.path.join(dir_bad, 'corrected')
    output_logfile_path = os.path.join(output_logfile_dir, f'corrected_{logfile}')
    # Check if the file is a regular file (not a directory)
    if os.path.isfile(input_filename):
        print("filenames: " + input_filename + " || " + output_logfile_path)
        sort_logfile(input_filename, output_logfile_path)
        player_count += 1
        a = 0
        
        player_pos_human = []
        timesteps = []
        player_pos_astro = []
        player_orientation_human = []
        player_orientation_astro = []
        object_positions = {

        }
        object_hold_states = {}
        object_identified_states = {}
        scores = []
        timelefts = []
        dep_balls = []

        # PROCESS SORTED DATA
        with open(output_logfile_path, 'r') as file:
            # Iterate through each line in the file
            for line in file:
                data = json.loads(line)
                if a<len(LAYOUTS): l = LAYOUTS[a]
                else: l = LAYOUTS[-1]

                if data['layout'] != l: 
                    if args.plot and data['layout'] == "level_three":
                        print("Plotting now...")

                        # Create a figure and axis
                        fig, ax = plt.subplots(figsize=(15, 15))

                        def update(frame):
                            ax.clear()
                            
                            # Plot human positions in brown
                            human_pos = player_pos_human[frame]
                            human_orientation = player_orientation_human[frame]
                            ax.scatter(human_pos[0], human_pos[1], c='brown', label='Human')
                            ax.quiver(human_pos[0], human_pos[1], human_orientation[0], human_orientation[1], angles='xy', scale_units='xy', scale=0.7, color='brown')
                            
                            # Plot astro positions in blu
                            astro_pos = player_pos_astro[frame]
                            astro_orientation = player_orientation_astro[frame]
                            ax.scatter(astro_pos[0], astro_pos[1], c='blue', label='Astro')
                            ax.quiver(astro_pos[0], astro_pos[1], astro_orientation[0], astro_orientation[1], angles='xy', scale_units='xy', scale=0.7, color='blue')
                            
                            # Plot object positions by color
                            for color, positions in object_positions.items():
                                color_map = {'green_1': 'green','green_2': 'green','green_3': 'green', 'yellow_1': 'orange', 'yellow_2': 'orange', 'yellow_3': 'orange', 'red_1': 'red', 'red_2': 'red', 'red_3': 'red'}
                                ax.scatter(positions[frame][0], positions[frame][1], c=color_map[color], label=f'{color.capitalize()} Objects')
                            
                            # Set grid limits and labels
                            ax.set_xlim(0, 15)
                            ax.set_ylim(0, 15)
                            ax.set_xticks(range(16))
                            ax.set_yticks(range(16))
                            ax.set_xlabel('X Position')
                            ax.set_ylabel('Y Position')
                            ax.set_title(f'Player {id} Cond:{COND} - {l} (Time: {timesteps[frame]})')
                            ax.grid(True)
                            
                            # Avoid duplicate legend entries
                            handles, labels = plt.gca().get_legend_handles_labels()
                            by_label = dict(zip(labels, handles))
                            ax.legend(by_label.values(), by_label.keys())

                        # Create animation
                        ani = animation.FuncAnimation(fig, update, frames=len(timesteps), repeat=False)

                        # Show the plot
                        plt.show()
                        input("cleck")
                    
                    trajectories_bad[l].append(player_pos_human)
                    total_time_bad[data['id']][l].append(calculate_game_duration(timesteps[0], timesteps[-1]))
                    process_mean_dist_to_astro(l,player_pos_human,player_pos_astro, COND, data['id'])
                    undep_list = []
                    id_list = []
                    unid_list = []
                    for obj in object_positions.keys():
                        if obj in dep_balls:
                            if True in object_identified_states[obj]:id_list.append(obj)
                            else: unid_list.append(obj)
                        else: undep_list.append(obj)
                    
                    identified_dep_balls_bad[data['id']][l].append(id_list)
                    unidentified_dep_balls_bad[data['id']][l].append(unid_list)
                    undep_balls_bad[data['id']][l].append(undep_list)

                    player_pos_human = []
                    timesteps = []
                    player_pos_astro = []
                    player_orientation_human = []
                    player_orientation_astro = []
                    object_positions = {

                    }
                    object_hold_states = {}
                    object_identified_states = {}
                    scores = []
                    timelefts = []
                    a+=1
                    dep_balls = []
                
                
                if data['id'] not in player_ids_bad:
                    player_ids_bad.append(data['id'])
                    time_spent_player_bad[data['id']] = {
                        "level_zero": {"red": 0, "yellow": 0, "green": 0},
                        "level_one": {"red": 0, "yellow": 0, "green": 0},
                        "level_two": {"red": 0, "yellow": 0, "green": 0},
                        "level_three": {"red": 0, "yellow": 0, "green": 0}
                    }
                    positions_holding_ball_bad[data['id']] = {
                    "level_zero": [],
                    "level_one": [],
                    "level_two": [],
                    "level_three": []
                    }
            
                    total_time_bad[data['id']] = {
                    "level_zero": [],
                    "level_one": [],
                    "level_two": [],
                    "level_three": []
                    }

                    mean_distance_to_astro_bad[data['id']] = {
                    "level_zero": [],
                    "level_one": [],
                    "level_two": [],
                    "level_three": []
                    }
                    identified_dep_balls_bad[data['id']] = {
                    "level_zero": [],
                    "level_one": [],
                    "level_two": [],
                    "level_three": []
                    }
                    unidentified_dep_balls_bad[data['id']] = {
                    "level_zero": [],
                    "level_one": [],
                    "level_two": [],
                    "level_three": []
                    }
                    undep_balls_bad[data['id']] = {
                    "level_zero": [],
                    "level_one": [],
                    "level_two": [],
                    "level_three": []
                    }

                    print(data['id'])
                # Extract player positions
                player_pos_human.append(data['players'][0]['position'])
                player_pos_astro.append(data['players'][1]['position'])
                player_orientation_human.append(data['players'][0]['orientation'])
                player_orientation_astro.append(data['players'][1]['orientation'])
                
                # Extract object information
                for obj in data['objects']:
                    obj_name = obj['name']
                    obj_pos = obj['position']
                    obj_hold_state = obj['hold_state']
                    obj_identified = obj['identified']
                    
                    # Store object positions based on their color
                    if obj_name not in object_positions.keys():
                        object_positions[obj_name] = []
                    object_positions[obj_name].append(obj_pos)
                    
                    # Store object hold states
                    if obj_name not in object_hold_states:
                        object_hold_states[obj_name] = []
                    object_hold_states[obj_name].append(obj_hold_state)
                    
                    # Store object identified states
                    if obj_name not in object_identified_states:
                        object_identified_states[obj_name] = []
                    object_identified_states[obj_name].append(obj_identified)
                
                # Extract score and timeleft
                scores.append(data['score'])
                timelefts.append(data['timeleft'])
                timesteps.append(data['time'])
                if data['players'][1]['held_object'] != None: dep_balls = data['players'][1]['held_object']
                else: dep_balls=[]

                #extract time holding ball
                timestamp, color = parse_time_balls(line)
                if data['players'][0]['held_object']: 
                    if data['players'][0]['position'] not in positions_holding_ball_bad[data['id']][data['layout']]:
                        positions_holding_ball_bad[data['id']][data['layout']].append(data['players'][0]['position'])

                if last_timestamp and last_color:
                    time_diff = (timestamp - last_timestamp).total_seconds()
                    time_spent_player_bad[data['id']][data['layout']][last_color] += time_diff
    
                last_timestamp = timestamp
                last_color = color


        trajectories_bad[l].append(player_pos_human)
        total_time_bad[data['id']][l].append(calculate_game_duration(timesteps[0], timesteps[-1]))
        process_mean_dist_to_astro(l,player_pos_human,player_pos_astro, COND, data['id'])
        print("LENS MEANS: ", len(mean_distance_to_astro_bad), len(mean_distance_to_astro_bad), len(mean_distance_to_astro_bad), len(mean_distance_to_astro_bad))
        undep_list = []
        id_list = []
        unid_list = []
        for obj in object_positions.keys():
            if obj in dep_balls:
                if True in object_identified_states[obj]:id_list.append(obj)
                else: unid_list.append(obj)
            else: undep_list.append(obj)
        
        identified_dep_balls_bad[data['id']][l].append(id_list)
        unidentified_dep_balls_bad[data['id']][l].append(unid_list)
        undep_balls_bad[data['id']][l].append(undep_list)

        #print("id balls: ", identified_dep_balls_bad)
        #print("unid balls: ", unidentified_dep_balls_bad)
        #print("undep balls: ", undep_balls_bad)
        #print("time holding ball: ", time_spent_bad)
    
    if args.plot:
        print("Plotting now...")

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(15, 15))

        def update(frame):
            ax.clear()
            
            # Plot human positions in brown
            human_pos = player_pos_human[frame]
            human_orientation = player_orientation_human[frame]
            ax.scatter(human_pos[0], human_pos[1], c='brown', label='Human')
            ax.quiver(human_pos[0], human_pos[1], human_orientation[0], human_orientation[1], angles='xy', scale_units='xy', scale=0.7, color='brown')
            
            # Plot astro positions in blu
            astro_pos = player_pos_astro[frame]
            astro_orientation = player_orientation_astro[frame]
            ax.scatter(astro_pos[0], astro_pos[1], c='blue', label='Astro')
            ax.quiver(astro_pos[0], astro_pos[1], astro_orientation[0], astro_orientation[1], angles='xy', scale_units='xy', scale=0.7, color='blue')
            
            # Plot object positions by color
            for color, positions in object_positions.items():
                color_map = {'green_1': 'green','green_2': 'green','green_3': 'green', 'yellow_1': 'orange', 'yellow_2': 'orange', 'yellow_3': 'orange', 'red_1': 'red', 'red_2': 'red', 'red_3': 'red'}
                ax.scatter(positions[frame][0], positions[frame][1], c=color_map[color], label=f'{color.capitalize()} Objects')
            
            # Set grid limits and labels
            ax.set_xlim(0, 15)
            ax.set_ylim(0, 15)
            ax.set_xticks(range(16))
            ax.set_yticks(range(16))
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_title(f'Player {id} Cond:{COND} - {l} (Time: {timesteps[frame]})')
            ax.grid(True)
            
            # Avoid duplicate legend entries
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(timesteps), repeat=False)

        # Show the plot
        plt.show()

print("player ids bad: ", player_ids_bad)
print("player count: ", player_count)
for player_id, levels in time_spent_player_bad.items():
    output = [f"{player_id}:"]
    for level, colors in levels.items():
        for color, time in colors.items():
            output.append(f"{level}_{color}: {time}:")
    print(" ".join(output))
input("STARTING PROCESSING TIME GOOD")
############ PROCESSING TIME GOOD ########################
'''for level in total_time_good:
    print("Level: ", level)
    for i in range(len(total_time_good[level])):
        print(f"Total time of game {player_ids_good[i]}: {total_time_good[level][i]:.2f} seconds")'''

for player_id, levels in total_time_good.items():
    output = [f"{player_id}:"]
    for level, times in levels.items():
        output.append(f"total_time_{level}: {times[0]:.2f}:")
    print(" ".join(output))


# Calculate mean times
# Print results
for player_id, levels in time_spent_player_good.items():
    output = [f"{player_id}:"]
    for level, colors in levels.items():
        for color, time in colors.items():
            output.append(f"{level}_{color}: {time}:")
    print(" ".join(output))

input("PROCESSING TIME BAD")
############ PROCESSING TIME BAD ########################

'''for level in total_time_bad:
    print("Level: ", level)
    for i in range(len(total_time_bad[level])):
        print(f"Total time of game {player_ids_bad[i]}: {total_time_bad[level][i]:.2f} seconds")'''


for player_id, levels in total_time_bad.items():
    output = [f"{player_id}:"]
    for level, times in levels.items():
        output.append(f"total_time_{level}: {times[0]:.2f}:")
    print(" ".join(output))

# Calculate mean times
print("BAD CONDITION")
# Print results
for player_id, levels in time_spent_player_bad.items():
    output = [f"{player_id}:"]
    for level, colors in levels.items():
        for color, time in colors.items():
            output.append(f"{level}_{color}: {time}:")
    print(" ".join(output))


input()
################################################################################################################


'''input("STARTING PROCESSING OF TRAJECTORIES.")

############ PROCESSING TRAJECTORIES GOOD ########################
trajectories = trajectories_good['level_three']
trajectories.extend(trajectories_bad['level_three'])
print(len(trajectories))
input()
# Preprocess the Data (Example: Normalization)
max_len = max(len(t) for t in trajectories)
trajectories = [np.pad(t, ((0, max_len - len(t)), (0, 0)), 'constant',  constant_values=t[-1]) for t in trajectories]
# Step 2: Feature Extraction (Optional: Here we use raw data)
# Flatten trajectories for clustering
flattened_trajectories = np.array([t.flatten() for t in trajectories])

# Step 3: Distance Calculation (Using DTW for example)
dist_matrix = np.zeros((len(trajectories), len(trajectories)))
for i, traj1 in enumerate(trajectories):
    for j, traj2 in enumerate(trajectories):
        distance, _ = fastdtw(traj1, traj2, dist=euclidean)
        dist_matrix[i, j] = distance

# Step 4: Clustering (Example: KMeans)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(flattened_trajectories)
kmeans = KMeans(n_clusters=10)
labels = kmeans.fit_predict(scaled_data)

# Step 5: Visualization
colors = ['r', 'g', 'b', 'c', 'm', 'k', 'y', 'darkred', 'purple', 'darkolivegreen']
plt.figure(figsize=(10, 8))
for i, traj in enumerate(trajectories):
    plt.plot(traj[:, 0], traj[:, 1], color=colors[labels[i]], label=f'Trajectory {i+1}')
plt.legend()
plt.title('Trajectories Clustering Good Condition')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()



############ PROCESSING TRAJECTORIES BAD########################
trajectories = trajectories_bad['level_three']
# Preprocess the Data (Example: Normalization)
max_len = max(len(t) for t in trajectories)
trajectories = [np.pad(t, ((0, max_len - len(t)), (0, 0)), 'constant',  constant_values=t[-1]) for t in trajectories]

# Step 2: Feature Extraction (Optional: Here we use raw data)
# Flatten trajectories for clustering
flattened_trajectories = np.array([t.flatten() for t in trajectories])

# Step 3: Distance Calculation (Using DTW for example)
dist_matrix = np.zeros((len(trajectories), len(trajectories)))
for i, traj1 in enumerate(trajectories):
    for j, traj2 in enumerate(trajectories):
        distance, _ = fastdtw(traj1, traj2, dist=euclidean)
        dist_matrix[i, j] = distance

# Step 4: Clustering (Example: KMeans)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(flattened_trajectories)
kmeans = KMeans(n_clusters=len(trajectories))
labels = kmeans.fit_predict(scaled_data)

# Step 5: Visualization
colors = ['r', 'g', 'b', 'c', 'm']
plt.figure(figsize=(10, 8))
for i, traj in enumerate(trajectories):
    plt.plot(traj[:, 0], traj[:, 1], color=colors[labels[i]], label=f'Trajectory {i+1}')
plt.legend()
plt.title('Trajectories Clustering Bad Condition')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()'''


input("STARTING PROCESSING OF DISTANCE TO ROBOT.")

############ PROCESSING DISTANCE TO ROBOT ########################

print("GOOD")
for player_id, levels in mean_distance_to_astro_good.items():
    output = [f"{player_id}:"]
    for level, mean in levels.items():
        output.append(f"mean_dist_{level}: {mean[0]:.2f}:")
    print(" ".join(output))

input("BAD")
for player_id, levels in mean_distance_to_astro_bad.items():
    output = [f"{player_id}:"]
    for level, mean in levels.items():
        output.append(f"mean_dist_{level}: {mean[0]:.2f}:")
    print(" ".join(output))


'''# Define levels and number of players
levels = list(mean_distance_to_astro_good.keys())
num_levels = len(levels)
num_players_good = len(mean_distance_to_astro_good[levels[0]])
num_players_bad = len(mean_distance_to_astro_bad[levels[0]])

# Calculate positions
width = 0.35  # Width of each bar
spacing = 1  # Space between different levels
positions_good = []
positions_bad = []

for i in range(num_levels):
    start = i * ((num_players_good + num_players_bad) * width + spacing)
    positions_good.extend([start + j * width for j in range(num_players_good)])
    positions_bad.extend([start + (num_players_good + j) * width for j in range(num_players_bad)])

# Flatten the distances
distances_good = [dist for level in levels for dist in mean_distance_to_astro_good[level]]
distances_bad = [dist for level in levels for dist in mean_distance_to_astro_bad[level]]

# Flatten the labels for the x-axis
labels_good = [f'{level.replace("_", " ").capitalize()} - P{i+1}' for level in levels for i in range(num_players_good)]
labels_bad = [f'{level.replace("_", " ").capitalize()} - P{i+1}' for level in levels for i in range(num_players_bad)]
labels = labels_good + labels_bad

# Plotting the data
fig, ax = plt.subplots(figsize=(15, 8))

ax.bar(positions_good, distances_good, width=width, color='blue', label='Good')
ax.bar(positions_bad, distances_bad, width=width, color='red', label='Bad')

# Setting the x-ticks
positions = positions_good + positions_bad
ax.set_xticks([pos + width/2 for pos in positions])
ax.set_xticklabels(labels, rotation=45, ha='right')

# Adding labels and title
ax.set_xlabel('Levels and Players')
ax.set_ylabel('Mean Distance')
ax.set_title('Mean Distance to Astro per Level and Player for Good and Bad Conditions')
ax.legend()

# Display the plot
plt.tight_layout()
plt.savefig(f'mean_distances_to_astro.png')
plt.show()'''

input("STARTING PROCESSING OF BALL STATISTICS.")

############ PROCESSING BALL STATISTICS ########################

print("LEN id_dep_good: ", len(identified_dep_balls_good))
print("LEN id_dep_bad: ", len(identified_dep_balls_bad))
print("LEN undep_good: ", len(undep_balls_good))
print("LEN undep_bad: ", len(undep_balls_bad))

print("GOOD CONDITION")

for player_id, levels in positions_holding_ball_good.items():
    output = [f"{player_id}:"]
    for level, positions in levels.items():
        output.append(f"pos_balls_{level}: {len(positions)}:")
    print(" ".join(output))

input()

print_ball_counts("Good")

'''print("IDD AND COLLECTED: ")
count_colors_by_level(identified_dep_balls_good)
print("NOT IDD AND COLLECTED: ")
count_colors_by_level(unidentified_dep_balls_good)
print("NOT COLLECTED: ")
count_colors_by_level(undep_balls_good)'''

print("BAD CONDITION")

for player_id, levels in positions_holding_ball_bad.items():
    output = [f"{player_id}:"]
    for level, positions in levels.items():
        output.append(f"pos_balls_{level}: {len(positions)}:")
    print(" ".join(output))

input()

print_ball_counts("Bad")
'''
print("IDD AND COLLECTED: ")
count_colors_by_level(identified_dep_balls_bad)
print("NOT IDD AND COLLECTED: ")
count_colors_by_level(unidentified_dep_balls_bad)
print("NOT COLLECTED: ")
count_colors_by_level(undep_balls_bad)'''

