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
import matplotlib.lines as mlines
import math
import pickle


cluster0_ids = ["5b7b22a5b5727c0001de1369","5d00bb3b51669b000136a986","5d02ed8f7a3c0f0015cd3230","5dafea4de40355001651fa2f","5e7bb46f580e3f0b33f82754","5f0da6848c9769018bb99f27","5fc7de54a68dc30770dad8c4","6019c3effc196e171c964e9e","60bc487bce7d279e68557d21","60eae4b46c198651395d4706","60f1ce3399bde1bc649825e9","60f32f4226cf00e90d5f7f2a","60fffea35327f2e456fac0d6","6109656bb28bc0fadcad52ca","611cd2b72cd5c01aeb3c6d8c","611cf4676874122c771e0c70","6130b0a83d0cfb9dda0f7b8e","613f442a2a07b0ffa0e17220","615b3bb442219baba00d0c71","62974e1b2e81d39409ee2781","64e76ad4ce316de314ca559b","64f72a267bed40bce1df18cb","6507355f07adea6bb9abb279","6595de8d3201bf1d41d9466c","6596bc0954d3bc04b7b9357a","660eaa43944e437daee5d3b8","664fac266d114b13daa8751d","66583199d6c1b95db128c4ee","666b06f0395fbff66097a1fc"]
cluster1_ids = ["578b98199bd3d70001e4f9bc","5d6a40c065a26e001731509a","5dcaca406f3f1679d7a65c67","5df1a3387caa1e0c69dca179","5e9ef7ee8de09c011aa656b6","5ea734aec721ae103213635a","5edd1c649e50c2a126f60b17","5f7c41f542eebd02c85a59ea","5f9dde49bcf5b5363fe6c6a3","602fc8d80c9b4ad3a35ce223","60f1d99a67bc1538a6dadc94","610173f93632b8b45c2130a7","61260d47007c8de7b40ca5b4","612bfd916d56dd67104c0a69","6136cb52db0f1cb2262bbf5d","6154e16c9500b0262bbcabaa","61715c627d1dd19a48cfeb91","650c3eaef6d3f3fc1349fa45","651c10ed5a2ace1639b27414","653d2bc468e530270a8d4c40","65abde27c2a42b82fabec539","65f33319737e19c6aad351b3","65fadedad737be57cd91c61f","6613f83d1a9cf059d0af91ec","6614daf007670261189c3e8a","66153c3cc57de8108716f452","6615c429a31a4c09456fec9d","662f624e7981b208ff1173ea","6630ec10efe9f6624eb24482","663b8816a6633f8069daade8","664ce375c75a5d05002a6ee7","6650a9190f37c9050d40e1e0","665d92789132bbfa9f3287c3","665facf2ba95668b3533d522","6666fda352745041d12def64","6672f004b99cab96498a3420","66745fa76bef7eae0db82ac6","66756b98b57dc3e151a38741","667650fe30eea4b3069143cc","667c0b3ce0d36fe18857cfa1","667c1343d8a7f6a0e22cc870"]
cluster2_ids = ["5c94bac30955b70012a521ce","5f5503435d41a489068ff50b","6088a7e22d5b98ef3f813a22","6098000de6e5af4368f40718","60bbce79ba111569d6d37efb","60f9caa879617464bc5df386","614f42e282f6218d9020c1a2","62ee4337445b044245a71185","65f3de969fe6519e23dff558","661fce887692b406ea5d9d26","665b7956b898d2f448616ea2"]
cluster0_lvl2_ids = ["578b98199bd3d70001e4f9bc","5b7b22a5b5727c0001de1369","5c94bac30955b70012a521ce","5d6a40c065a26e001731509a","5df1a3387caa1e0c69dca179","5e7bb46f580e3f0b33f82754","5e9ef7ee8de09c011aa656b6","5ea734aec721ae103213635a","5f7c41f542eebd02c85a59ea","60f32f4226cf00e90d5f7f2a","611cd2b72cd5c01aeb3c6d8c","61260d47007c8de7b40ca5b4","612bfd916d56dd67104c0a69","614f42e282f6218d9020c1a2","6154e16c9500b0262bbcabaa","61715c627d1dd19a48cfeb91","62ee4337445b044245a71185","64f72a267bed40bce1df18cb","651c10ed5a2ace1639b27414","653d2bc468e530270a8d4c40","65abde27c2a42b82fabec539","65f33319737e19c6aad351b3","660eaa43944e437daee5d3b8","6614daf007670261189c3e8a","6615c429a31a4c09456fec9d","662f624e7981b208ff1173ea","6630ec10efe9f6624eb24482","663b8816a6633f8069daade8","664ce375c75a5d05002a6ee7","6650a9190f37c9050d40e1e0","665facf2ba95668b3533d522","6666fda352745041d12def64","666b06f0395fbff66097a1fc","667c1343d8a7f6a0e22cc870"]
cluster1_lvl2_ids = ["5d00bb3b51669b000136a986","5d02ed8f7a3c0f0015cd3230","5dafea4de40355001651fa2f","5dcaca406f3f1679d7a65c67","5edd1c649e50c2a126f60b17","5f0da6848c9769018bb99f27","5f5503435d41a489068ff50b","5f9dde49bcf5b5363fe6c6a3","5fc7de54a68dc30770dad8c4","6019c3effc196e171c964e9e","602fc8d80c9b4ad3a35ce223","6088a7e22d5b98ef3f813a22","6098000de6e5af4368f40718","60bbce79ba111569d6d37efb","60bc487bce7d279e68557d21","60eae4b46c198651395d4706","60f1ce3399bde1bc649825e9","60f1d99a67bc1538a6dadc94","60f9caa879617464bc5df386","60fffea35327f2e456fac0d6","610173f93632b8b45c2130a7","6109656bb28bc0fadcad52ca","611cf4676874122c771e0c70","6130b0a83d0cfb9dda0f7b8e","6136cb52db0f1cb2262bbf5d","613f442a2a07b0ffa0e17220","615b3bb442219baba00d0c71","62974e1b2e81d39409ee2781","64e76ad4ce316de314ca559b","6507355f07adea6bb9abb279","650c3eaef6d3f3fc1349fa45","6595de8d3201bf1d41d9466c","6596bc0954d3bc04b7b9357a","65f3de969fe6519e23dff558","65fadedad737be57cd91c61f","6613f83d1a9cf059d0af91ec","66153c3cc57de8108716f452","661fce887692b406ea5d9d26","664fac266d114b13daa8751d","66583199d6c1b95db128c4ee","665b7956b898d2f448616ea2","665d92789132bbfa9f3287c3","6672f004b99cab96498a3420","66745fa76bef7eae0db82ac6","66756b98b57dc3e151a38741","667650fe30eea4b3069143cc","667c0b3ce0d36fe18857cfa1"]
chosen_ids = ['5d6a40c065a26e001731509a', '5df1a3387caa1e0c69dca179', '5f7c41f542eebd02c85a59ea', '602fc8d80c9b4ad3a35ce223', '612bfd916d56dd67104c0a69', '6136cb52db0f1cb2262bbf5d', '66153c3cc57de8108716f452', '662f624e7981b208ff1173ea', '667650fe30eea4b3069143cc', '667c1343d8a7f6a0e22cc870']




LAYOUTS = ["level_zero", "level_one", "level_two", "level_three"]

# Initialize lists to store the extracted information
trajectories_good = {"level_zero":[],"level_one":[],"level_two":[],"level_three":[] }
ball_embeddings = {"level_zero":[],"level_one":[],"level_two":[],"level_three":[] } # ball embeddings: [id_dep, id_undep, unid_dep, unid_undep]
mean_distance_to_astro_good = {}
idd_balls = {"level_zero":0,"level_one":0,"level_two":0,"level_three":0}
identified_dep_balls_good = {}
identified_undep_balls_good = {}
unidentified_dep_balls_good = {}
unidentified_undep_balls_good = {}
undep_balls_good = {}
total_time_good = {}
game_scores={}
nr_object_positions = {}

trajectories_bad = {"level_zero":[],"level_one":[],"level_two":[],"level_three":[] }
mean_distance_to_astro_bad = {}

all_trajectories = {}

identified_dep_balls_bad = {}
identified_undep_balls_bad = {}
fake_id_dep_balls_bad = {}
fake_id_undep_balls_bad = {}
unidentified_dep_balls_bad = {}
unidentified_undep_balls_bad = {}
undep_balls_bad = {}
total_time_bad = {}

player_ids_good = []
player_ids_bad = []

player_pos_human = []
timesteps = []
player_pos_astro = []
player_orientation_human = []
player_orientation_astro = []
object_positions = {}
object_hold_states = {}
object_identified_states = {}
scores = []
timelefts = []
dep_balls = []
time_spent_player_good = {}
positions_holding_ball_good = {}
all_pos_holding_ball_good = {}
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
            if player['position'][0] == -1: player['position'][0] = 0
            if player['position'][1] == -1: player['position'][1] = 0
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
    for ball in ball_list:
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
                id_undep_counts = count_balls(identified_undep_balls_good[player_id][level])
                unid_dep_counts = count_balls(unidentified_dep_balls_good[player_id][level])
                unid_undep_counts = count_balls(unidentified_undep_balls_good[player_id][level])

                ball_embeddings[level].append([sum(id_dep_counts.values()),sum(id_undep_counts.values()), sum(unid_dep_counts.values()), sum(unid_undep_counts.values())])

                #undep_counts = count_balls(undep_balls_good[player_id][level])
                output.append(f"{level}_id_dep_red: {id_dep_counts['red']}: {level}_id_dep_green: {id_dep_counts['green']}: {level}_id_dep_yellow: {id_dep_counts['yellow']}:")
                output.append(f"{level}_id_undep_red: {id_undep_counts['red']}: {level}_id_undep_green: {id_undep_counts['green']}: {level}_id_undep_yellow: {id_undep_counts['yellow']}:")
                output.append(f"{level}_unid_dep_red: {unid_dep_counts['red']}: {level}_unid_dep_green: {unid_dep_counts['green']}: {level}_unid_dep_yellow: {unid_dep_counts['yellow']}:")
                output.append(f"{level}_unid_undep_red: {unid_undep_counts['red']}: {level}_unid_undep_green: {unid_undep_counts['green']}: {level}_unid_undep_yellow: {unid_undep_counts['yellow']}:")
                #output.append(f"{level}_undep_red: {undep_counts['red']}: {level}_undep_green: {undep_counts['green']}: {level}_undep_yellow: {undep_counts['yellow']}")
            print(" ".join(output))     
    
    elif COND == "Bad":
        for player_id, levels in identified_dep_balls_bad.items():
            output = [f"{player_id}:"]
            for level in levels.keys():
                id_dep_counts = count_balls(identified_dep_balls_bad[player_id][level])
                id_undep_counts = count_balls(identified_undep_balls_bad[player_id][level])
                unid_dep_counts = count_balls(unidentified_dep_balls_bad[player_id][level])
                unid_undep_counts = count_balls(unidentified_undep_balls_bad[player_id][level])
                fakeid_dep_counts = count_balls(fake_id_dep_balls_bad[player_id][level])
                fakeid_undep_counts = count_balls(fake_id_undep_balls_bad[player_id][level])
                
                ball_embeddings[level].append([sum(id_dep_counts.values())+sum(fakeid_dep_counts.values()),sum(id_undep_counts.values())+sum(fakeid_undep_counts.values()), sum(unid_dep_counts.values()), sum(unid_undep_counts.values())])
                
                #undep_counts = count_balls(undep_balls_bad[player_id][level])
                output.append(f"{level}_id_dep_red: {id_dep_counts['red']}: {level}_id_dep_green: {id_dep_counts['green']}: {level}_id_dep_yellow: {id_dep_counts['yellow']}:")
                output.append(f"{level}_id_undep_red: {id_undep_counts['red']}: {level}_id_undep_green: {id_undep_counts['green']}: {level}_id_undep_yellow: {id_undep_counts['yellow']}:")
                output.append(f"{level}_unid_dep_red: {unid_dep_counts['red']}: {level}_unid_dep_green: {unid_dep_counts['green']}: {level}_unid_dep_yellow: {unid_dep_counts['yellow']}:")
                output.append(f"{level}_unid_undep_red: {unid_undep_counts['red']}: {level}_unid_undep_green: {unid_undep_counts['green']}: {level}_unid_undep_yellow: {unid_undep_counts['yellow']}:")
                output.append(f"{level}_fakeid_dep_red: {fakeid_dep_counts['red']}: {level}_fakeid_dep_green: {fakeid_dep_counts['green']}: {level}_fakeid_dep_yellow: {fakeid_dep_counts['yellow']}:")
                output.append(f"{level}_fakeid_undep_red: {fakeid_undep_counts['red']}: {level}_fakeid_undep_green: {fakeid_undep_counts['green']}: {level}_fakeid_undep_yellow: {fakeid_undep_counts['yellow']}:")
                #output.append(f"{level}_undep_red: {undep_counts['red']}: {level}_undep_green: {undep_counts['green']}: {level}_undep_yellow: {undep_counts['yellow']}")
            
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

def euclidean_distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def check_false_identification(file):

    threshold_time = 6  # seconds
    astro_proximity = {}
    already_id = []
    current_layout = None

    with open(output_logfile_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            game_id = data['id']
            layout = data['layout']
            current_time = datetime.strptime(data['time'], '%Y-%m-%d %H:%M:%S.%f')

            if current_layout and current_layout != layout:
                # When layout changes, the level has ended
                for ob in obj_list:
                    if astro['held_object']:
                        # if not deposited
                        if ob['name'] not in astro['held_object']:
                            # if not identified fr
                            if not ob['identified']:
                                for obj_name, info in astro_proximity.items():
                                    #if fake id - add to fake_id_undep and remove from unid_undep
                                    if obj_name == ob['name'] and info['fake_id']: 
                                        fake_id_undep_balls_bad[game_id][current_layout].append(obj_name)
                                        unidentified_undep_balls_bad[game_id][current_layout].remove(obj_name)

                # Reset astro_proximity for the new layout
                astro_proximity.clear()
                already_id = []

            current_layout = layout

            astro = next(player for player in data['players'] if player['name'] == 'astro')
            astro_pos = astro['position']
            obj_list = data['objects']
            for obj in obj_list:
                obj_name = obj['name']
                obj_pos = obj['position']
                identified = obj['identified']
                deposited = obj['hold_state'] == 2
                if not identified and obj_name not in already_id:
                    distance = euclidean_distance(astro_pos, obj_pos)
                    if distance <= math.sqrt(2):
                        if obj_name not in astro_proximity:
                            astro_proximity[obj_name] = {'start_time': current_time, 'fake_id': False}
                        else:
                            if not astro_proximity[obj_name]['fake_id']:
                                duration = current_time - astro_proximity[obj_name]['start_time']
                                if duration.total_seconds() > threshold_time:
                                    astro_proximity[obj_name]['fake_id'] = True
                    else:
                        if obj_name in astro_proximity and not astro_proximity[obj_name]['fake_id']:
                            del astro_proximity[obj_name]
                    
                    if deposited and obj_name in astro_proximity and astro_proximity[obj_name]['fake_id']: 
                        if obj_name not in fake_id_dep_balls_bad[game_id][current_layout]: 
                            fake_id_dep_balls_bad[game_id][current_layout].append(obj_name)
                            unidentified_dep_balls_bad[game_id][current_layout].remove(obj_name)
                else: 
                    already_id.append(obj_name)
                    if obj_name in astro_proximity: del astro_proximity[obj_name]    

        idd_balls[current_layout] += len(set(already_id))
        # Checking deposited and undeposited balls for last layout
        for obj in obj_list:
            if astro['held_object']:
            # if not deposited
                if obj['name'] not in astro['held_object']:
                    # if not identified fr
                    if not obj['identified']:
                        for obj_name, info in astro_proximity.items():
                            #if fake id - add to fake_id_undep and remove from unid_undep
                            if obj_name == obj['name'] and info['fake_id']: 
                                fake_id_undep_balls_bad[game_id][current_layout].append(obj_name)
                                unidentified_undep_balls_bad[game_id][current_layout].remove(obj_name)
"""    print("fake id: ", fake_id_dep_balls_bad[game_id], fake_id_undep_balls_bad[game_id])
    print("id:", identified_dep_balls_bad[game_id], identified_undep_balls_bad[game_id])
    print("unid:", unidentified_dep_balls_bad[game_id], unidentified_undep_balls_bad[game_id])
    input()"""

def plot_nr_positions_with_ball(data_good,data_bad):
    levels = ['level_zero', 'level_one', 'level_two', 'level_three']
    for level in levels:
        plt.figure(figsize=(10, 6))
        
        for key in data_good:
            timesteps = list(range(len(data_good[key][level])))
            plt.plot(timesteps, data_good[key][level], color='blue')

        for key in data_bad:
            timesteps = list(range(len(data_bad[key][level])))
            plt.plot(timesteps, data_bad[key][level], color='red')

        red_line = mlines.Line2D([], [], color='red', label='Bad Condition')
        blue_line = mlines.Line2D([], [], color='blue', label='Good Condition')
        
        plt.legend(handles=[red_line, blue_line])
        
        plt.title(f'{level}: # Positions Holding a Ball During Game')
        plt.xlabel('Timesteps')
        plt.ylabel('# Positions')
        plt.grid(True)
        plt.show()

#### PROCESSING GOOD CONDITION

COND = "Good"

# Ensure the directory path ends with a '/'
if not dir_good.endswith('/'):
    dir_good += '/'

files = os.listdir(dir_good)
files.sort()

# Iterate over all files in the directory
for logfile in files:
    input_filename = os.path.join(dir_good, logfile)
    output_logfile_dir = os.path.join(dir_good, 'corrected')
    output_logfile_path = os.path.join(output_logfile_dir, f'corrected_{logfile}')
    
    # Check if the file is a regular file (not a directory)
    if os.path.isfile(input_filename):
        #print("filenames: " + input_filename + " || " + output_logfile_path)
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
                if a == 0: print(data['id'])
                if a<len(LAYOUTS): l = LAYOUTS[a]
                else: l = LAYOUTS[-1]
                
                if data['layout'] != l: 
                    if args.plot and (l == "level_two"):
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

                    

                    trajectories_good[l].append(player_pos_human)
                    all_trajectories[data['id']][l] = trajectories_good[l][-1]
                    process_mean_dist_to_astro(l,player_pos_human,player_pos_astro, COND, data['id'])
                    total_time_good[data['id']][l].append(calculate_game_duration(timesteps[0], timesteps[-1]))
                    undep_list = []
                    id_dep = []
                    unid_dep = []
                    id_undep = []
                    unid_undep = []
                    for obj in object_positions.keys():
                        if obj in dep_balls:
                            if True in object_identified_states[obj]:id_dep.append(obj)
                            else: unid_dep.append(obj)
                        else:
                            if True in object_identified_states[obj]:id_undep.append(obj)
                            else: unid_undep.append(obj) 
                            undep_list.append(obj)
                    
                    identified_dep_balls_good[data['id']][l].extend(id_dep)
                    unidentified_dep_balls_good[data['id']][l].extend(unid_dep)
                    identified_undep_balls_good[data['id']][l].extend(id_undep)
                    unidentified_undep_balls_good[data['id']][l].extend(unid_undep)
                    undep_balls_good[data['id']][l].extend(undep_list)

                    game_scores[data['id']][l].append(scores[-1])
    

                    nr_id_balls = 0
                    for obj in object_positions.keys():
                        if True in object_identified_states[obj]: nr_id_balls += 1
                                        
                    idd_balls[l] += nr_id_balls


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
                    all_pos_holding_ball_good[data['id']] = {
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
                    identified_undep_balls_good[data['id']] = {
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
                    unidentified_undep_balls_good[data['id']] = {
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
                    game_scores[data['id']] = {
                    "level_zero": [],
                    "level_one": [],
                    "level_two": [],
                    "level_three": []
                    }
                    all_trajectories[data['id']] = {
                    "level_zero": [],
                    "level_one": [],
                    "level_two": [],
                    "level_three": []
                    }
                    nr_object_positions[data['id']] = {
                    "level_zero": {},
                    "level_one": {},
                    "level_two": {},
                    "level_three": {}
                    } 

                    
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
                    
                    # Populate the 'level one' dictionary with lengths of arrays
                    for color, positions in object_positions.items():                       
                        s = set(tuple(i) for i in positions)
                        nr_object_positions[data['id']][l][color] = len(s)

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

                #extract positions holding ball
                if data['players'][0]['held_object']: 
                    if data['players'][0]['position'] not in positions_holding_ball_good[data['id']][data['layout']]:
                        positions_holding_ball_good[data['id']][data['layout']].append(data['players'][0]['position'])
                all_pos_holding_ball_good[data['id']][data['layout']].append(len(positions_holding_ball_good[data['id']][data['layout']]))
    
                if last_timestamp and last_color:
                    time_diff = (timestamp - last_timestamp).total_seconds()
                    time_spent_player_good[data['id']][data['layout']][last_color] += time_diff

                last_timestamp = timestamp
                last_color = color
        
        trajectories_good[l].append(player_pos_human)
        all_trajectories[data['id']][l] = trajectories_good[l][-1]
        total_time_good[data['id']][l].append(calculate_game_duration(timesteps[0], timesteps[-1]))
        process_mean_dist_to_astro(l,player_pos_human,player_pos_astro, COND, data['id'])
        #print(len(mean_distance_to_astro_good), len(mean_distance_to_astro_good), len(mean_distance_to_astro_good), len(mean_distance_to_astro_good))
        undep_list = []
        id_dep = []
        unid_dep = []
        id_undep = []
        unid_undep = []
        for obj in object_positions.keys():
            if obj in dep_balls:
                if True in object_identified_states[obj]:id_dep.append(obj)
                else: unid_dep.append(obj)
            else:
                if True in object_identified_states[obj]:id_undep.append(obj)
                else: unid_undep.append(obj) 
                undep_list.append(obj)
        
        identified_dep_balls_good[data['id']][l].extend(id_dep)
        unidentified_dep_balls_good[data['id']][l].extend(unid_dep)
        identified_undep_balls_good[data['id']][l].extend(id_undep)
        unidentified_undep_balls_good[data['id']][l].extend(unid_undep)
        undep_balls_good[data['id']][l].append(undep_list)
        game_scores[data['id']][l].append(scores[-1])
        nr_id_balls = 0
        for obj in object_positions.keys():
            if True in object_identified_states[obj]: nr_id_balls += 1
        
        idd_balls[l] += nr_id_balls

        if args.plot and l == "level_two":
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


file = open('game_scores_good.pickle', 'wb')
pickle.dump(game_scores, file)
file.close()

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
game_scores={}
scores = []
timelefts = []
dep_balls = []
time_spent_player_bad = {}
positions_holding_ball_bad = {}
all_pos_holding_ball_bad = {}
last_timestamp = None
last_color = None
# Ensure the directory path ends with a '/'
if not dir_bad.endswith('/'):
    dir_bad += '/'


files = os.listdir(dir_bad)
files.sort()
# Iterate over all files in the directory
for logfile in files:
    input_filename = os.path.join(dir_bad, logfile)
    output_logfile_dir = os.path.join(dir_bad, 'corrected')
    output_logfile_path = os.path.join(output_logfile_dir, f'corrected_{logfile}')
    # Check if the file is a regular file (not a directory)
    if os.path.isfile(input_filename):
        #print("filenames: " + input_filename + " || " + output_logfile_path)
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
                if a == 0: print(data['id'])
                if a<len(LAYOUTS): l = LAYOUTS[a]
                else: l = LAYOUTS[-1]

                if data['layout'] != l: 
                    if args.plot and l == "level_two":
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

                    

                    trajectories_bad[l].append(player_pos_human)
                    all_trajectories[data['id']][l] = trajectories_bad[l][-1]
                    total_time_bad[data['id']][l].append(calculate_game_duration(timesteps[0], timesteps[-1]))
                    process_mean_dist_to_astro(l,player_pos_human,player_pos_astro, COND, data['id'])
                    undep_list = []
                    id_dep = []
                    unid_dep = []
                    id_undep = []
                    unid_undep = []
                    for obj in object_positions.keys():
                        if obj in dep_balls:
                            if True in object_identified_states[obj]:id_dep.append(obj)
                            else: unid_dep.append(obj)
                        else:
                            if True in object_identified_states[obj]:id_undep.append(obj)
                            else: unid_undep.append(obj) 
                            undep_list.append(obj)
                    
                    identified_dep_balls_bad[data['id']][l].extend(id_dep)
                    unidentified_dep_balls_bad[data['id']][l].extend(unid_dep)
                    identified_undep_balls_bad[data['id']][l].extend(id_undep)
                    unidentified_undep_balls_bad[data['id']][l].extend(unid_undep)
                    undep_balls_bad[data['id']][l].extend(undep_list)
                    game_scores[data['id']][l].append(scores[-1])
                    nr_id_balls = 0
                    for obj in object_positions.keys():
                        if True in object_identified_states[obj]: nr_id_balls += 1
                    
                    
                    idd_balls[l] += nr_id_balls

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
                    all_pos_holding_ball_bad[data['id']] = {
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
                    identified_undep_balls_bad[data['id']] = {
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
                    unidentified_undep_balls_bad[data['id']] = {
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
                    fake_id_dep_balls_bad[data['id']] = {
                    "level_zero": [],
                    "level_one": [],
                    "level_two": [],
                    "level_three": []
                    }
                    fake_id_undep_balls_bad[data['id']] = {
                    "level_zero": [],
                    "level_one": [],
                    "level_two": [],
                    "level_three": []
                    }
                    game_scores[data['id']] = {
                    "level_zero": [],
                    "level_one": [],
                    "level_two": [],
                    "level_three": []
                    }
                    all_trajectories[data['id']] = {
                    "level_zero": [],
                    "level_one": [],
                    "level_two": [],
                    "level_three": []
                    }
                    
                    nr_object_positions[data['id']] = {
                    "level_zero": {},
                    "level_one": {},
                    "level_two": {},
                    "level_three": {}
                    } 

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
                    
                    # Populate the 'level one' dictionary with lengths of arrays
                    for color, positions in object_positions.items():                       
                        s = set(tuple(i) for i in positions)
                        nr_object_positions[data['id']][l][color] = len(s)

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
                
                all_pos_holding_ball_bad[data['id']][data['layout']].append(len(positions_holding_ball_bad[data['id']][data['layout']]))

                if last_timestamp and last_color:
                    time_diff = (timestamp - last_timestamp).total_seconds()
                    time_spent_player_bad[data['id']][data['layout']][last_color] += time_diff
    
                last_timestamp = timestamp
                last_color = color

        trajectories_bad[l].append(player_pos_human)
        all_trajectories[data['id']][l] = trajectories_bad[l][-1]
        total_time_bad[data['id']][l].append(calculate_game_duration(timesteps[0], timesteps[-1]))

        process_mean_dist_to_astro(l,player_pos_human,player_pos_astro, COND, data['id'])
        #print("LENS MEANS: ", len(mean_distance_to_astro_bad), len(mean_distance_to_astro_bad), len(mean_distance_to_astro_bad), len(mean_distance_to_astro_bad))
        undep_list = []
        id_dep = []
        unid_dep = []
        id_undep = []
        unid_undep = []
        for obj in object_positions.keys():
            if obj in dep_balls:
                if True in object_identified_states[obj]:id_dep.append(obj)
                else: unid_dep.append(obj)
            else:
                if True in object_identified_states[obj]:id_undep.append(obj)
                else: unid_undep.append(obj) 
                undep_list.append(obj)
        
        nr_id_balls = 0
        for obj in object_positions.keys():
            if True in object_identified_states[obj]: nr_id_balls += 1
        

        idd_balls[l] += nr_id_balls

        identified_dep_balls_bad[data['id']][l].extend(id_dep)
        unidentified_dep_balls_bad[data['id']][l].extend(unid_dep)
        identified_undep_balls_bad[data['id']][l].extend(id_undep)
        unidentified_undep_balls_bad[data['id']][l].extend(unid_undep)
        undep_balls_bad[data['id']][l].extend(undep_list)
        game_scores[data['id']][l].append(scores[-1])

        check_false_identification(file)

    
    if args.plot and data['layout'] == "level_two":
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

#print("Idd bals: ", idd_balls)

file = open('game_scores_bad.pickle', 'wb')
pickle.dump(game_scores, file)
file.close()
plot_nr_positions_with_ball(all_pos_holding_ball_good, all_pos_holding_ball_bad)

file = open('all_trajectories.pickle', 'wb')
pickle.dump(all_trajectories, file)
file.close()

input("STARTING PROCESSING TIME GOOD")
############ PROCESSING TIME GOOD ########################
'''for level in total_time_good:
    print("Level: ", level)
    for i in range(len(total_time_good[level])):
        print(f"Total time of game {player_ids_good[i]}: {total_time_good[level][i]:.2f} seconds")'''

file = open('total_time_good.pickle', 'wb')
pickle.dump(total_time_good, file)
file.close()

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


file = open('total_time_bad.pickle', 'wb')
pickle.dump(total_time_bad, file)
file.close()

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


################################################################################################################

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

plt.plot()
plt.title()
file = open('positions_holding_ball_good.pickle', 'wb')
pickle.dump(positions_holding_ball_good, file)
file.close()

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

file = open('positions_holding_ball_bad.pickle', 'wb')
pickle.dump(positions_holding_ball_bad, file)
file.close()

id_list = ['5dafea4de40355001651fa2f','602fc8d80c9b4ad3a35ce223', '6088a7e22d5b98ef3f813a22','66153c3cc57de8108716f452', '6672f004b99cab96498a3420', '66756b98b57dc3e151a38741']
for id in id_list:
    print(id)
    print(identified_dep_balls_bad[id])
    print(identified_undep_balls_bad[id])
    print(unidentified_dep_balls_bad[id])
    print(unidentified_undep_balls_bad[id])
    print(fake_id_dep_balls_bad[id])
    print(fake_id_undep_balls_bad[id])
print_ball_counts("Bad")
file = open('ball_embeddings.pickle', 'wb')
pickle.dump(ball_embeddings, file)
file.close()
'''
print("IDD AND COLLECTED: ")
count_colors_by_level(identified_dep_balls_bad)
print("NOT IDD AND COLLECTED: ")
count_colors_by_level(unidentified_dep_balls_bad)
print("NOT COLLECTED: ")
count_colors_by_level(undep_balls_bad)'''

