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
import csv
import copy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import itertools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib.patches import Patch

LAYOUTS = ["level_zero", "level_one", "level_two", "level_three"]
OR_OFFSET = [[-1,0], [1,0], [0,-1], [0,1]]
chosen_ids = ['5dcaca406f3f1679d7a65c67', '5e9ef7ee8de09c011aa656b6', '5ea734aec721ae103213635a', '5f9dde49bcf5b5363fe6c6a3', '610173f93632b8b45c2130a7', '6154e16c9500b0262bbcabaa', '650c3eaef6d3f3fc1349fa45', '65abde27c2a42b82fabec539', '65f33319737e19c6aad351b3', '65fadedad737be57cd91c61f', '6613f83d1a9cf059d0af91ec', '6614daf007670261189c3e8a', '6615c429a31a4c09456fec9d', '6650a9190f37c9050d40e1e0', '665facf2ba95668b3533d522', '66756b98b57dc3e151a38741']

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

def cluster_no_reduction(X, num_clusters):
    ### CLUSTERING WITH K MEANS

    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    print(f"{num_clusters} CLUSTERS")
    kmeans.fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    print("Labels: ", labels)
    # Print Centers with custom formatting
    print("Centers:")
    formatted_centers = []
    for center in centers:
        formatted_center = [
            round(val, 6) if 1000 > val > 0.001 else float(f"{val:.6e}")
            for val in center
        ]
        formatted_centers.append(formatted_center)
    
    print(formatted_centers)
    
    # Initialize a list to store clusters
    clusters = [[] for _ in range(num_clusters)]

    for i, lb in enumerate(labels):
        key = list(MDMT_score_sorted.keys())[i]
        clusters[lb].append(MDMT_score_sorted[key]['Label'])

    # Print all clusters
    for i in range(num_clusters):
        print(f"Cluster {i}: ", clusters[i])

def detect_actions(p_pos1, r_pos1, p_or1, r_or1, hold_state1, id_state1, p_pos2, r_pos2, p_or2, r_or2, hold_state2, id_state2):
    #UP = 0 DOWN = 1 LEFT = 2 RIGHT = 3 INTERACT = 4 STAY = 5 IDENTIFY = 6
    def determine_movement(position1, position2):
        if position2[0] > position1[0]:
            return 1
        elif position2[0] < position1[0]:
            return 0
        elif position2[1] > position1[1]:
            return 3
        elif position2[1] < position1[1]:
            return 2
        return 5
    
    def determine_orientation(new_or):
        if new_or == [1,0]:
            return 1
        elif new_or == [-1,0]:
            return 0
        elif new_or == [0,1]:
            return 3
        elif new_or == [0,-1]:
            return 2        
        return 5

    def determine_interaction(hold_states1, hold_states2):
        if hold_states1 != hold_states2:
            return 4
        return 5

    player_position1, robot_position1 = p_pos1, r_pos1
    player_position2, robot_position2 = p_pos2, r_pos2
    
    player_orientation1, robot_orientation1 = p_or1, r_or1
    player_orientation2, robot_orientation2 = p_or2, r_or2
    
    player_hold_states1, player_hold_states2 = hold_state1, hold_state2
    robot_id_states1, robot_id_states2 = id_state1, id_state2

    # Determine player actions
    action_player = determine_movement(player_position1, player_position2)
    if action_player == 5 and player_orientation1 != player_orientation2:
        action_player = determine_orientation(player_orientation2)
    if action_player == 5:
        action_player = determine_interaction(player_hold_states1, player_hold_states2)

    # Determine robot actions
    action_robot = determine_movement(robot_position1, robot_position2)
    if action_robot == 5 and robot_orientation1 != robot_orientation2:
        action_robot = determine_orientation(robot_orientation2)

    if robot_id_states1 != robot_id_states2:
        action_robot = 6
    
    return action_player, action_robot

def plot_heatmap(positions_visited, title):
    # Initialize a 15x15 grid with zeros
    grid_size = 15
    grid = np.zeros((grid_size, grid_size))

    # Populate the grid with visit counts
    for position in positions_visited:
        x = int(position[0])
        y = int(position[1])
        grid[x, y] += 1

    normalized_data = (grid - np.min(grid)) / (np.max(grid) - np.min(grid))
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(normalized_data, cmap='hot_r', interpolation='nearest', origin='lower')
    # Add grid lines
    plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)
    plt.xticks(np.arange(-0.5, grid_size, 1), [])
    plt.yticks(np.arange(-0.5, grid_size, 1), [])

    plt.colorbar(label='Visit Count')
    plt.title(f'Heatmap of {title}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.gca().invert_yaxis()  # Optional: invert y-axis to have (0,0) at the bottom-left
    
    plt.show()

def process_position_or(player_pos, robot_pos, player_or, robot_or):
    #UP = 0 DOWN = 1 LEFT = 2 RIGHT = 3 INTERACT = 4 STAY = 5 IDENTIFY = 6

    p_player_pos_x = convert_to_binary(player_pos[0], '04b')
    p_player_pos_y = convert_to_binary(player_pos[1], '04b')

    p_player_pos_x.extend(p_player_pos_y)
    p_player_pos = copy.copy(p_player_pos_x)

    p_robot_pos_x = convert_to_binary(robot_pos[0], '04b')
    p_robot_pos_y = convert_to_binary(robot_pos[1], '04b')

    p_robot_pos_x.extend(p_robot_pos_y)
    p_robot_pos = copy.copy(p_robot_pos_x)

    vector = [0,0,0,0]
    if player_or == [0, 0]: vector[0] = 1
    else: vector[OR_OFFSET.index(player_or)] = 1

    p_player_or = vector

    vector = [0,0,0,0]
    if robot_or == [0, 0]: vector[0] = 1
    else: vector[OR_OFFSET.index(robot_or)] = 1

    p_robot_or = vector

    return p_player_pos, p_robot_pos, p_player_or, p_robot_or

def convert_to_binary(number, type):
    binary_string = format(number, type)
    return [int(digit) for digit in binary_string]

def normalize_raw_data_length(raw_data,lvl):

    def find_closest_entry(entries, target_timestamp):
        # Find the entry with the smallest difference in timestamp
        closest_entry = min(entries, key=lambda entry: abs(entry[-1] - target_timestamp))
        return closest_entry
    
    init_timestep = raw_data[0][-1]
    final_timestep = raw_data[-1][-1]
    time_interval = (final_timestep - init_timestep)/int(mean_levels[lvl])
    normalized_raw_data = []
    for i in range(0, int(mean_levels[lvl])):
        timestep = i*time_interval
        #search for timestep closest to value timestep
        entry = find_closest_entry(raw_data, init_timestep + timestep)
        normalized_raw_data.append(entry)

    return normalized_raw_data

def compare_centroids(centroid_array):
    """
    Compare the values at each index across different cluster centroids
    and print which cluster has the maximum value for each index.
    
    Parameters:
    centroid_array (numpy.ndarray): An array of shape (N, 42) where N is the number of clusters.
    """
    num_clusters, num_features = centroid_array.shape

    # Iterate through each index (0 to 41, representing the 42 features)
    for index in range(num_features):
        # Extract the values for the current index from each cluster
        values_at_index = centroid_array[:, index]
        
        # Find the cluster with the maximum value at this index
        max_cluster = np.argmax(values_at_index)
        max_value = values_at_index[max_cluster]

        # Output the result for this index
        print(f"Index {index + 1}: Cluster {max_cluster + 1} has the largest value ({max_value})")    

def plot_clusters(data, lvl):

    global level2_clusters, level3_3clusters, level3_clusters
    
    tsnes = []
    cluster_labels = []

    if args.section_game:
        for d in data:
            print("SHAPE: ", d.shape)
            flattened_data = d.reshape(d.shape[0], -1)
            print("FLAT SHAPE: ", flattened_data.shape)
            tsnes.append(tsne.fit_transform(flattened_data))
            cluster_labels.append(kmeans.fit_predict(flattened_data))
            print("CLUSTER LABELS")
            input(cluster_labels[-1])
    else:
        flattened_data = data.reshape(data.shape[0], -1)
        print("FLAT SHAPE: ", flattened_data.shape)
        tsnes.append(tsne.fit_transform(flattened_data))
        cluster_labels.append(kmeans.fit_predict(flattened_data))
        print("CLUSTER LABELS")
        input(cluster_labels[-1])
        
    if lvl == 2: level2_clusters = copy.copy(cluster_labels)
    elif lvl == 3: level3_clusters = copy.copy(cluster_labels)

    '''    
    #Find the PCA space centroids of each cluster
    cluster_centroids_pca = np.array([flattened_data[cluster_labels == i].mean(axis=0) for i in np.unique(cluster_labels)])

    # Use inverse PCA to map these centroids back to the original flattened space
    original_space_centroids = pca.inverse_transform(cluster_centroids_pca)  
    print("SHAPE CENTROID ORIGINAL: ", original_space_centroids.shape)
    # Reshape each centroid back to the original 3D shape 
    original_shape_centroids = original_space_centroids.reshape(-1, data.shape[1],data.shape[2]) 
    print("SHAPE CENTROID ORIGINAL: ", original_shape_centroids.shape)

    mean_centroid_array = []
    np.set_printoptions(threshold=np.inf)
    for cluster_idx, centroid in enumerate(original_shape_centroids):
        mean_centroid = np.mean(centroid, axis=0)
        mean_centroid_array.append(mean_centroid)
        print(f"\nCluster {cluster_idx + 1} Centroid:")
        #print(np.array2string(np.array(reverse_standardization(mean_centroid, means[LAYOUTS[lvl]],stdevs[LAYOUTS[lvl]])), formatter={'float_kind': lambda x: f"{x:.1f}"}))  # Print rounded values for readability
    '''

    # Cluster label
    for i,tsne_results in enumerate(tsnes):
        
        # Define unique clusters and corresponding colors
        unique_clusters = np.unique(cluster_labels[i])
        colors = plt.cm.plasma(np.linspace(0, 1, len(unique_clusters)))

        # Create a dictionary for cluster labels and their corresponding colors
        legend_elements = [Patch(facecolor=colors[i], edgecolor='k', label=f'Cluster {unique_clusters[i]}') for i in range(len(unique_clusters))]
        
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=cluster_labels[i], cmap='plasma', edgecolors='k', s=50)
        plt.title(f"t-SNE of Level {lvl} Part {i} - Clusters")
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(handles=legend_elements, title="Cluster label", loc='best')
        plt.show()

        # MDMT label
        print("mdmt: " , mdmt_labels)
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=mdmt_labels, cmap='coolwarm_r', edgecolors='k', s=50)
        plt.title(f"t-SNE of Level {lvl} Part {i} - MDMT Labels")
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.show()

        #Condition Label
        print("Condition label:", condition_labels)
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=condition_labels, cmap='coolwarm_r', edgecolors='k', s=50)
        plt.title(f"t-SNE of Level {lvl} Part {i} - Condition Labels")
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.show()

    a = input("Do you want to see more clusters? y/n: ")
    if a == "y":
        cluster_labels2 = []
        kmeans2 = KMeans(n_clusters=5, random_state=42)
        if args.section_game:
            for d in data:
                print("SHAPE: ", d.shape)
                flattened_data = d.reshape(d.shape[0], -1)
                print("FLAT SHAPE: ", flattened_data.shape)
                cluster_labels2.append(kmeans2.fit_predict(flattened_data))
                input(cluster_labels2)
        else:
            flattened_data = data.reshape(data.shape[0], -1)
            print("FLAT SHAPE: ", flattened_data.shape)
            cluster_labels2.append(kmeans2.fit_predict(flattened_data))
            print("CLUSTER LABELS")
            input(cluster_labels2[-1])  
        
        if lvl == 3: level3_3clusters = copy.copy(cluster_labels2)
        # Cluster label
        for i,tsne_results in enumerate(tsnes):
            # Define unique clusters and corresponding colors
            unique_clusters2 = np.unique(cluster_labels2[i])
            colors = plt.cm.plasma(np.linspace(0, 1, len(unique_clusters2)))

            # Create a dictionary for cluster labels and their corresponding colors
            legend_elements = [Patch(facecolor=colors[i], edgecolor='k', label=f'Cluster {unique_clusters2[i]}') for i in range(len(unique_clusters2))]
            plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=cluster_labels2[i], cmap='plasma', edgecolors='k', s=50)
            plt.title(f"t-SNE of Level {lvl} Part {i} - Clusters2")
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            # Adding legend
            plt.legend(handles=legend_elements, title="Cluster label", loc='best')
            plt.show()
            
            # MDMT label
            print("mdmt: " , mdmt_labels)
            plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=mdmt_labels, cmap='coolwarm_r', edgecolors='k', s=50)
            plt.title(f"t-SNE of Level {lvl} Part {i} - MDMT Labels")
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.show()

            #Condition Label
            print("Condition label:", condition_labels)
            plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=condition_labels, cmap='coolwarm_r', edgecolors='k', s=50)
            plt.title(f"t-SNE of Level {lvl} Part {i} - Condition Labels")
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.show()

def standardize_entry(entry,means,stdevs,lvl):
    std_entry = []
    for i in range(0,len(entry)):
        if lvl == 0 and i > 3 and i < 20: 
            std_entry.append(entry[i])
            continue #do not normalize ball hold state and id state and orientation
        elif lvl != 0 and i > 3 and i < 32: 
            std_entry.append(entry[i])
            continue #do not normalize ball hold state and id state and orientation
        if stdevs[i] == 0: std_entry.append(0)
        else: std_entry.append(entry[i]-means[i]/stdevs[i])
    return std_entry

def reverse_standardization(entry,means,stdevs):
    reversed_entry = []
    input(entry)
    for i in range(0,len(entry)):
        if stdevs[i] == 0: reversed_entry.append(0)
        else: reversed_entry.append(round((entry[i]*stdevs[i])+means[i],1))
    return reversed_entry

def classifier(data_lvl2, data_lvl3):
    max_steps = min(data_lvl2[0].shape[1], data_lvl3[0].shape[1])
    flattened_data = []
    cluster_labels = []
    for i in range(len(data_lvl2)):
        data = copy.copy(data_lvl2[i][:, :max_steps, :])
        flattened_data.append(data.reshape(data.shape[0], -1))
        cluster_labels.append(kmeans.fit_predict(flattened_data[-1]))

    flat_data = np.concatenate(flattened_data,axis=0)
    flat_clusters = np.concatenate(cluster_labels,axis=0)
    print("FLAT DATA: ", flat_data.shape)
    print("FLAT CLUSTER: ", flat_clusters.shape)
    input()
    # Prepare training data
    X_train = flat_data

    ### CLASSIFY BY CLUSTER
    y_train = flat_clusters

    # Split data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Train a Logistic Regression classifier
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)

    # Optionally, evaluate the classifier on the validation set
    accuracy = clf.score(X_val, y_val)
    print(f"Validation Accuracy: {accuracy}")

    #DATA FROM LEVEL 3
    flattened_data = []
    for i in range(len(data_lvl3)):
        print("SHAPE: ", data_lvl3[i].shape)
        new_data = copy.copy(data_lvl3[i][:, :max_steps, :])
        print("SHAPE: ", new_data.shape)
        flattened_data.append(new_data.reshape(new_data.shape[0], -1))
        print("SHAPE: ", flattened_data[-1].shape)

    new_flat_data = np.concatenate(flattened_data,axis=0)
    print("SHAPE: ", new_flat_data.shape)
    input()
    # Predict cluster labels for the new data
    new_labels = clf.predict(new_flat_data)

    print(new_labels[:new_data.shape[0]])
    print(len(new_labels[:new_data.shape[0]]))
    print(new_labels[new_data.shape[0]:new_data.shape[0]*2])
    print(len(new_labels[new_data.shape[0]:new_data.shape[0]*2]))
    print(new_labels[new_data.shape[0]*2:])
    print(len(new_labels[new_data.shape[0]*2:]))

raw_data = {}
std_raw_data = {}
normalized_raw_data = {}

# Initialize the dictionary
MDMT_score = {}

# Create argument parser
parser = argparse.ArgumentParser(description='Parse JSON log files in a directory.')
parser.add_argument('--logdir', dest='logdir', type=str, required=True, help='Path to directory containing JSON log files')
parser.add_argument('--std_way', dest='std_way', type=int, required=True, help='Way to standardize data: 1.binary encoding most of it + std for score and timestep; 2. std everything equally; 3. std based on timestep')
parser.add_argument('--section_game', dest='section_game', action='store_true', help='Cluster games on sections of 3')
parser.add_argument('--classifier', dest='classifier', action='store_true', help='Train classifier')
# Parse command line arguments
args = parser.parse_args()
logfile_dir = args.logdir
all_dir = logfile_dir + "/All/"
player_ids = []
player_ids_bad = ['60bbce79ba111569d6d37efb','65f3de969fe6519e23dff558','5f9dde49bcf5b5363fe6c6a3','5d02ed8f7a3c0f0015cd3230','667c0b3ce0d36fe18857cfa1','6088a7e22d5b98ef3f813a22','66153c3cc57de8108716f452','62974e1b2e81d39409ee2781','66745fa76bef7eae0db82ac6','5f0da6848c9769018bb99f27','6019c3effc196e171c964e9e','60f1d99a67bc1538a6dadc94','5f5503435d41a489068ff50b','602fc8d80c9b4ad3a35ce223','6130b0a83d0cfb9dda0f7b8e','6507355f07adea6bb9abb279','6613f83d1a9cf059d0af91ec','613f442a2a07b0ffa0e17220','5fc7de54a68dc30770dad8c4','663b8816a6633f8069daade8','6596bc0954d3bc04b7b9357a','667650fe30eea4b3069143cc','65fadedad737be57cd91c61f','6630ec10efe9f6624eb24482','5dafea4de40355001651fa2f','6098000de6e5af4368f40718','5d00bb3b51669b000136a986','5dcaca406f3f1679d7a65c67',	'665b7956b898d2f448616ea2','650c3eaef6d3f3fc1349fa45','6672f004b99cab96498a3420','611cd2b72cd5c01aeb3c6d8c','6136cb52db0f1cb2262bbf5d','665d92789132bbfa9f3287c3','661fce887692b406ea5d9d26','60fffea35327f2e456fac0d6','66756b98b57dc3e151a38741','5b7b22a5b5727c0001de1369','5e7bb46f580e3f0b33f82754','60f32f4226cf00e90d5f7f2a']
player_ids_good = ['664fac266d114b13daa8751d','60f9caa879617464bc5df386','6650a9190f37c9050d40e1e0','6595de8d3201bf1d41d9466c','61715c627d1dd19a48cfeb91','578b98199bd3d70001e4f9bc','61260d47007c8de7b40ca5b4','6109656bb28bc0fadcad52ca','651c10ed5a2ace1639b27414','5c94bac30955b70012a521ce','5d6a40c065a26e001731509a','612bfd916d56dd67104c0a69','6615c429a31a4c09456fec9d','60bc487bce7d279e68557d21','615b3bb442219baba00d0c71','653d2bc468e530270a8d4c40','5e9ef7ee8de09c011aa656b6','64f72a267bed40bce1df18cb','65f33319737e19c6aad351b3','5ea734aec721ae103213635a','5edd1c649e50c2a126f60b17','611cf4676874122c771e0c70','64e76ad4ce316de314ca559b','65abde27c2a42b82fabec539','66583199d6c1b95db128c4ee','667c1343d8a7f6a0e22cc870','665facf2ba95668b3533d522','664ce375c75a5d05002a6ee7','60f1ce3399bde1bc649825e9','60eae4b46c198651395d4706','662f624e7981b208ff1173ea','614f42e282f6218d9020c1a2','5f7c41f542eebd02c85a59ea','6614daf007670261189c3e8a','6666fda352745041d12def64','5df1a3387caa1e0c69dca179','62ee4337445b044245a71185','610173f93632b8b45c2130a7','6154e16c9500b0262bbcabaa','666b06f0395fbff66097a1fc','660eaa43944e437daee5d3b8']

files = os.listdir(all_dir)
files.sort()
# Iterate over all files in the directory
for logfile in files:
    input_filename = os.path.join(all_dir, logfile)
    output_logfile_dir = os.path.join(all_dir, 'corrected')
    output_logfile_path = os.path.join(output_logfile_dir, f'corrected_{logfile}')
    
    # Check if the file is a regular file (not a directory)
    if os.path.isfile(input_filename):
        print("filenames: " + input_filename + " || " + output_logfile_path)
        sort_logfile(input_filename, output_logfile_path)
        a = 0


        # PROCESS SORTED DATA
        with open(output_logfile_path, 'r') as file:
            # Iterate through each line in the file

            for line in file:
                raw_data_vector = [] 
                hold_state = []
                id_state = []
                data = json.loads(line)
                if data['id'] not in chosen_ids: continue

                if data['id'] not in player_ids:
                    player_ids.append(data['id'])
                    raw_data[data['id']] = {
                        "level_zero": [],
                        "level_one": [],
                        "level_two": [],
                        "level_three": []
                    }    
                    std_raw_data[data['id']] = {
                        "level_zero": [],
                        "level_one": [],
                        "level_two": [],
                        "level_three": []
                    }                  

                ## BUILDING RAW DATA VECTOR : [player pos, astro pos, player or, astro or, ball1 state, ball2 state ..., ball1 id, ball2 id, ...,  player action, astro action, score, timestep]  
                
                if args.std_way == 1:
                    # Extract player positions
                    p_player_pos, p_robot_pos, p_player_or, p_robot_or = process_position_or([int(x) for x in data['players'][0]['position']],[int(x) for x in data['players'][1]['position']], data['players'][0]['orientation'],data['players'][1]['orientation'])

                    raw_data_vector.extend(p_player_pos)
                    raw_data_vector.extend(p_robot_pos)

                    raw_data_vector.extend(p_player_or)
                    raw_data_vector.extend(p_robot_or)
                    
                    # Extract object information
                    for obj in data['objects']:
                        raw_data_vector.append(obj['hold_state'])
                        hold_state.append(obj['hold_state'])
                    
                    for obj in data['objects']:
                        if obj['identified']: raw_data_vector.append(1)
                        else: raw_data_vector.append(0)
                        id_state.append(raw_data_vector[-1])

                    ## EXTRACT ACTIONS FROM DATA
                    if len(raw_data[data['id']][data['layout']]) > 0:
                        action_player, action_robot = detect_actions(prev_pos_or[0], prev_pos_or[1], prev_pos_or[2], prev_pos_or[3], prev_hold_state, prev_id_state, data['players'][0]['position'],data['players'][1]['position'],data['players'][0]['orientation'],data['players'][1]['orientation'], hold_state, id_state)
                    else: 
                        action_player = 5
                        action_robot = 5

                    action_player_vector = convert_to_binary(action_player, '03b')
                    raw_data_vector.extend(action_player_vector)
                    action_robot_vector = convert_to_binary(action_robot, '03b')
                    raw_data_vector.extend(action_robot_vector)
                    raw_data_vector.append(data['score'])
                    raw_data_vector.append(data['timeleft'])
                    #raw_data_vector.append(datetime.strptime(data['time'], "%Y-%m-%d %H:%M:%S.%f").timestamp())
                    raw_data[data['id']][data['layout']].append(raw_data_vector)
                    prev_hold_state = copy.copy(hold_state)
                    prev_id_state = copy.copy(id_state)
                    prev_pos_or = [data['players'][0]['position'],data['players'][1]['position'],data['players'][0]['orientation'],data['players'][1]['orientation']]
                    
                elif args.std_way == 2 or args.std_way == 3:
                    
                    raw_data_vector.extend([int(x) for x in data['players'][0]['position']])
                    #raw_data_vector.extend([int(x) for x in data['players'][1]['position']])

                    raw_data_vector.extend(data['players'][0]['orientation'])
                    #raw_data_vector.extend(data['players'][1]['orientation'])
        
                    
                    # Extract object information
                    for obj in data['objects']:
                        vector = np.zeros(3)
                        vector[obj['hold_state']] = 1
                        raw_data_vector.extend(vector)
                        hold_state.append(obj['hold_state'])

                    for obj in data['objects']:
                        if obj['identified']: raw_data_vector.append(1)
                        else: raw_data_vector.append(0)
                        id_state.append(raw_data_vector[-1])

                    ## EXTRACT ACTIONS FROM DATA
                    if len(raw_data[data['id']][data['layout']]) > 0:
                        action_player, action_robot = detect_actions(prev_pos_or[0], prev_pos_or[1], prev_pos_or[2], prev_pos_or[3], prev_hold_state, prev_id_state, data['players'][0]['position'],data['players'][1]['position'],data['players'][0]['orientation'],data['players'][1]['orientation'], hold_state, id_state)
                    else: 
                        action_player = 5
                        action_robot = 5

                    raw_data_vector.append(action_player)
                    #raw_data_vector.append(action_robot)
                    raw_data_vector.append(data['score'])
                    #raw_data_vector.append(data['timeleft'])
                    #raw_data_vector.append(datetime.strptime(data['time'], "%Y-%m-%d %H:%M:%S.%f").timestamp())
                    
                    raw_data[data['id']][data['layout']].append(raw_data_vector)
                    prev_hold_state = copy.copy(hold_state)
                    prev_id_state = copy.copy(id_state)
                    prev_pos_or = [data['players'][0]['position'],data['players'][1]['position'],data['players'][0]['orientation'],data['players'][1]['orientation']]

# Read the CSV file
with open('MDMT.csv', mode='r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        if row['ID'] in chosen_ids:
            MDMT_score[row['ID']] = float(row['Mean'])
            label = 0 if 0 <= float(row['Mean']) <= 5.4 else 1 if 5.4 < float(row['Mean']) <= 7 else None
            if label is not None:  # Ensure the mean_value falls within the valid range
                MDMT_score[row['ID']] = {'Mean': float(row['Mean']), 'Label': label}



MDMT_score_sorted = {id_: MDMT_score[id_] for id_ in player_ids if id_ in MDMT_score}
len_lvl0 = []
len_lvl1 = []
len_lvl2 = []
len_lvl3 = []   
level2_clusters = None
level3_clusters = None   
level3_3clusters = None           

for id, item in raw_data.items():
    len_lvl0.append(len(item['level_zero']))
    len_lvl1.append(len(item['level_one']))
    len_lvl2.append(len(item['level_two']))
    len_lvl3.append(len(item['level_three']))

mean_lvl0 = np.average(len_lvl0)
mean_lvl1 = np.average(len_lvl1)
mean_lvl2 = np.average(len_lvl2)
mean_lvl3 = np.average(len_lvl3)
mean_levels = [mean_lvl0,mean_lvl1,mean_lvl2,mean_lvl3]

print(mean_lvl0,mean_lvl1,mean_lvl2, mean_lvl3)
normalized_raw_data = {
    "level_zero": [],
    "level_one": [],
    "level_two": [],
    "level_three": []
}    

mdmt_labels = []
condition_labels = []

print("STANDARDIZING DATA")

def get_shape(nested_list):
    if isinstance(nested_list, list):
        # Get the length of the current list and recursively get the shape of the first element
        return [len(nested_list)] + get_shape(nested_list[0])
    else:
        # Base case: if it's not a list, return an empty list
        return []

#STANDARDIZE DATA
if args.std_way==1 or args.std_way==3:
    std_raw_data = copy.copy(raw_data)

elif args.std_way==2:
    means = {
        "level_zero": [],
        "level_one": [],
        "level_two": [],
        "level_three": []
    }      
    stdevs = {
        "level_zero": [],
        "level_one": [],
        "level_two": [],
        "level_three": []
    } 
    for l in LAYOUTS:
        level_data = list(itertools.chain.from_iterable(p_data[l] for p_data in raw_data.values()))
        print(get_shape(level_data)) 
        for i in range(0,len(level_data[0])):
            means[l].append(np.mean(np.array(level_data)[:,i]))
            stdevs[l].append(np.std(np.array(level_data)[:,i]))

    for id, data in tqdm(raw_data.items()):
        for l in LAYOUTS:
            for entry in data[l]:
                std_entry = standardize_entry(entry,means[l],stdevs[l],LAYOUTS.index(l))
                std_raw_data[id][l].append(std_entry)

print("NORMALIZING DATA LENGTH")
# NORMALIZE DATA LENGTH
for id, data in std_raw_data.items():
    normalized_raw_data_level0 = normalize_raw_data_length(data['level_zero'],0)
    normalized_raw_data_level1 = normalize_raw_data_length(data['level_one'],1)
    normalized_raw_data_level2 = normalize_raw_data_length(data['level_two'],2)
    normalized_raw_data_level3 = normalize_raw_data_length(data['level_three'],3)
    normalized_raw_data['level_zero'].append(normalized_raw_data_level0)
    normalized_raw_data['level_one'].append(normalized_raw_data_level1)
    normalized_raw_data['level_two'].append(normalized_raw_data_level2)
    normalized_raw_data['level_three'].append(normalized_raw_data_level3)
    mdmt_labels.append(MDMT_score_sorted[id]['Label'])
    if id in player_ids_bad: condition_labels.append(0)
    elif id in player_ids_good: condition_labels.append(1)

normalized_data = {
        "level_zero": np.empty(3, dtype=object),
        "level_one": np.empty(3, dtype=object),
        "level_two": np.empty(3, dtype=object),
        "level_three": np.empty(3, dtype=object)
    } 

if args.std_way ==1 or args.std_way==2:
    if args.section_game:
        print("SECTIONING GAMES")
        for l in LAYOUTS:
            first_third_length = np.array(normalized_raw_data[l]).shape[1]//3
            second_third_length = first_third_length + np.array(normalized_raw_data[l]).shape[1]//3
            normalized_data[l][0] = np.array(normalized_raw_data[l])[:, :first_third_length, :]
            normalized_data[l][1] = np.array(normalized_raw_data[l])[:, first_third_length:second_third_length, :]
            normalized_data[l][2] = np.array(normalized_raw_data[l])[:, second_third_length:, :]


    else:
        normalized_data['level_zero'] = np.array(normalized_raw_data['level_zero'])
        normalized_data['level_one'] = np.array(normalized_raw_data['level_one'])
        normalized_data['level_two'] = np.array(normalized_raw_data['level_two'])
        normalized_data['level_three']= np.array(normalized_raw_data['level_three'])

elif args.std_way==3:
    std_raw_data = {
        "level_zero": [],
        "level_one": [],
        "level_two": [],
        "level_three": []
        }  
    means = {}      
    stdevs = {} 
    for l, p_data in normalized_raw_data.items():
        means[l]=np.array(np.mean(p_data, axis=0))
        stdevs[l]=np.array(np.std(p_data, axis=0))

    for l, data in normalized_raw_data.items():
        for entries in data:
            std_vector = []
            for i, entry in enumerate(entries):
                std_entry = standardize_entry(entry,means[l][i],stdevs[l][i],LAYOUTS.index(l))
                std_vector.append(std_entry)
            std_raw_data[l].append(std_vector)
    if args.section_game:
        print("SECTIONING GAMES")
        for l in LAYOUTS:
            first_third_length = np.array(std_raw_data[l]).shape[1]//3
            second_third_length = first_third_length*2
            normalized_data[l][0] = np.array(std_raw_data[l])[:, :first_third_length, :]
            normalized_data[l][1] = np.array(std_raw_data[l])[:, first_third_length:second_third_length, :]
            normalized_data[l][2] = np.array(std_raw_data[l])[:, second_third_length:, :]

    else:
        normalized_data['level_zero'] = np.array(std_raw_data['level_zero'])
        normalized_data['level_one'] = np.array(std_raw_data['level_one'])
        normalized_data['level_two'] = np.array(std_raw_data['level_two'])
        normalized_data['level_three'] = np.array(std_raw_data['level_three'])

pca = PCA(n_components=50)
tsne = TSNE(n_components=2, random_state=42)
kmeans = KMeans(n_clusters=2, random_state=42)  # Assuming 2 clusters

if args.classifier: classifier(normalized_data["level_two"], normalized_data["level_three"])  
else: 
    for i,l in enumerate(LAYOUTS): 
        if l == "level_two" or l == "level_three": plot_clusters(normalized_data[l], i)


'''
### HEATMAPS

positions_visited_level2_cluster0 = []
positions_visited_level2_cluster1 = []
positions_visited_level3_cluster0 = []
positions_visited_level3_cluster1 = []
positions_visited_level3_3cluster0 = []
positions_visited_level3_3cluster1 = []
positions_visited_level3_3cluster2 = []


i = 0

for id, item in raw_data.items():
    for instance in item['level_two']:
        if level2_clusters[0][i] == 0:
            positions_visited_level2_cluster0.append([instance[0], instance[1]])
        elif level2_clusters[0][i] == 1:
            positions_visited_level2_cluster1.append([instance[0], instance[1]])

    for instance in item['level_three']:
        if level3_clusters[0][i] == 0:
            positions_visited_level3_cluster0.append([instance[0], instance[1]])
        elif level3_clusters[0][i] == 1:
            positions_visited_level3_cluster1.append([instance[0], instance[1]])
        
        if level3_3clusters[0][i] == 0:
            positions_visited_level3_3cluster0.append([instance[0], instance[1]])
        elif level3_3clusters[0][i] == 1:
            positions_visited_level3_3cluster1.append([instance[0], instance[1]])
        elif level3_3clusters[0][i] == 2:
            positions_visited_level3_3cluster2.append([instance[0], instance[1]])
    
    i += 1

plot_heatmap(positions_visited_level2_cluster0, "Level 2 cluster 0")
plot_heatmap(positions_visited_level2_cluster1, "Level 2 cluster 1")
plot_heatmap(positions_visited_level3_cluster0, "Level 3 cluster 0")
plot_heatmap(positions_visited_level3_cluster1, "Level 3 cluster 1")
plot_heatmap(positions_visited_level3_3cluster0, "Level 3 3 clusters 0")
plot_heatmap(positions_visited_level3_3cluster1, "Level 3 3 clusters 1")
plot_heatmap(positions_visited_level3_3cluster2, "Level 3 3 clusters 2")
'''
