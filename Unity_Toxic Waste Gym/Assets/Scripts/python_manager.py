import UnityEngine as ue
import json

f = open(f'/home/anavc/Toxic_Waste_Gym/Unity_Toxic Waste Gym/Assets/state.json')
 
# returns JSON object as 
# a dictionary
data = json.load(f)
if data['command'] == 'new_lvl':
    # Load next scene
    sceneName = data['data']['layout']
elif data['command'] == 'new_state':

    human_x, human_y = data['data']['players']['human']['position']
    human_ox,human_oy = data['data']['players']['human']['orientation']
    human_holding = data['data']['players']['human']['ball in hand']

    robot_x, robot_y = data['data']['players']['robot']['position']
    robot_ox,robot_oy = data['data']['players']['robot']['orientation']
    robot_holding = data['data']['players']['robot']['ball in hand']

    health = data['data']['players']['human']['health']
    score = data['data']['score']

    all_objects = ue.Object.FindObjectsOfType(ue.GameObject)

    for go in all_objects:

        # HUMAN PLAYER
        if go.name == "Player": # coordinate conversion: x_unity = y_backend; y_unity = 14 - x_backend
            #Movement and Rotation
            go.GetComponent("Human_Movement").moveOrRotate(ue.Vector3(human_y,14-human_x,0), ue.Vector2(human_ox,human_oy))
            
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
            if data['data']['objects'][go.name]['status'] == "disposed": ue.Object.Destroy(go)

        # HEALTH AND SCORE
        if go.name == "GameHandler": go.GetComponent("GameHandler").update_Score_Health(score, health)


    for i in data['data']['objects']:
        ue.Debug.Log(i)

    ue.Debug.Log(data['data']['score'])


# Closing file
f.close()

