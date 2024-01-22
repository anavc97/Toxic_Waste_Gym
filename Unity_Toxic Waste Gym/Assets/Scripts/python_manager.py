import UnityEngine as ue
import json

f = open(f'/home/anavc/Toxic_Waste_Gym/Unity_Toxic Waste Gym/Assets/state.json')
 
# returns JSON object as 
# a dictionary
data = json.load(f)

human_x, human_y = data['players']['human']['position']
human_o = data['players']['human']['orientation']
human_holding = data['players']['human']['ball in hand']

robot_x, robot_y = data['players']['robot']['position']
robot_o = data['players']['robot']['orientation']
robot_holding = data['players']['robot']['ball in hand']

all_objects = ue.Object.FindObjectsOfType(ue.GameObject)
for go in all_objects:

    # HUMAN PLAYER
    if go.name == "Player":
        #check if position and rotation are different

        #Movement
        go.GetComponent("Human_Movement").Print("Hello") # Get component of movement script to move and rotate character
        #go.transform.position = ue.Vector3(human_x,human_y,0)
        
        #Ball in hand
        if human_holding:
            #Change sprite to holding ball
            ue.Debug.Log("Human is holding ball")


    # ROBOT 
    if go.name == "Robot":
        #Movement
        go.GetComponent("Human_Movement").Print("Hello") # Get component of movement script to move and rotate character
        #go.transform.position = ue.Vector3(human_x,human_y,0)
        
        #Ball in hand
        if robot_holding:
            #Change sprite to holding ball
            ue.Debug.Log("Robot is holding ball(s)")
    
    # BALLS
    if "ball" in go.name:
        ue.Debug.Log(go)
        if data['objects'][go.name]['status'] == "disposed": ue.Object.Destroy(go)

for i in data['objects']:
    ue.Debug.Log(i)

ue.Debug.Log(data['score'])


# Closing file
f.close()

