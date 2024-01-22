import UnityEngine as ue
import json

f = open(f'/home/anavc/Toxic_Waste_Gym/Unity_Toxic Waste Gym/Assets/state.json')
 
# returns JSON object as 
# a dictionary
data = json.load(f)

human_x, human_y = data['players']['human']['position']
human_ox,human_oy = data['players']['human']['orientation']
human_holding = data['players']['human']['ball in hand']

robot_x, robot_y = data['players']['robot']['position']
robot_ox,robot_oy = data['players']['robot']['orientation']
robot_holding = data['players']['robot']['ball in hand']

health = data['players']['human']['health']
score = data['score']

all_objects = ue.Object.FindObjectsOfType(ue.GameObject)

for go in all_objects:

    # HUMAN PLAYER
    if go.name == "Player":          
        #Movement and Rotation
        go.GetComponent("Human_Movement").moveOrRotate(ue.Vector3(human_x,human_y,0), ue.Vector2(human_ox,human_oy))
        
        #Ball in hand
        if human_holding:
            #Change sprite to holding ball
            ue.Debug.Log("Human is holding ball")


    # ROBOT 
    if go.name == "Robot":
        #Movement
        #go.GetComponent("Robot_Movement").moveOrRotate(ue.Vector3(robot_x,robot_y,0), ue.Vector2(robot_ox,robot_oy)) 
        
        #Ball in hand
        if robot_holding:
            #Change sprite to holding ball
            ue.Debug.Log("Robot is holding ball(s)")
    
    # BALLS
    if "ball" in go.name:
        ue.Debug.Log(go)
        if data['objects'][go.name]['status'] == "disposed": ue.Object.Destroy(go)

    # HEALTH AND SCORE
    if go.name == "GameHandler": go.GetComponent("GameHandler").update_Score_Health(score, health)


for i in data['objects']:
    ue.Debug.Log(i)

ue.Debug.Log(data['score'])


# Closing file
f.close()

