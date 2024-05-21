import json
import time

def update_positions(data):
    human_position = data["players"]["human"]["position"]
    robot_position = data["players"]["robot"]["position"]
    health = data["players"]["human"]["health"] 
    score = data["score"]

    human_position[0] -= 1  # Decrease x coordinate for human
    robot_position[1] += 1  # Increase y coordinate for robot
    health = health - 0.1
    score += 1

    # Update the positions in the data
    data["players"]["human"]["position"] = human_position
    data["players"]["robot"]["position"] = robot_position
    data["players"]["human"]["health"] = health
    data["score"] = score

    return data


def main():
    # Read JSON file
    with open("state.json", "r") as file:
        json_data = json.load(file)
    
    i = 5

    while i>0:
        # Update positions
        updated_data = update_positions(json_data)

        # Write updated data back to the JSON file
        with open("state.json", "w") as file:
            json.dump(updated_data, file, indent=2)

        # Wait for 1 second
        time.sleep(1)
        i -= 1


if __name__ == "__main__":
    main()