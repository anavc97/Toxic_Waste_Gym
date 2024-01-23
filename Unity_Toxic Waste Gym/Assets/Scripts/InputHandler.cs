using System.Collections;
using System.Collections.Generic;
using UnityEngine;
//using System.Text.Json;
//using System.Text.Json.Serialization;
//using Newtonsoft.Json;


public class NewBehaviourScript : MonoBehaviour
{
    [System.Serializable]
    public class Action
    {
        public string command;
        public int id;
        public int data; //0=up, 1=down, 2=left, 3=right, 4=interact
    }
    
    private string filePath =  Application.dataPath + "/human_action.json";
    private bool actionExecuted = false;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {       
        Action action = new Action();
        if(Mathf.Abs(Input.GetAxisRaw("Horizontal")) == 1f)
        {
            action = new Action
            {
                command = "Player action",
                id = 1,
                data = (int)(2.5 + (Input.GetAxisRaw("Horizontal") / 2)) //2=left 3=right
            };
            actionExecuted = true;
        }
        else if(Mathf.Abs(Input.GetAxisRaw("Vertical")) == 1f)
        {
            action = new Action
            {
                command = "Player action",
                id = 1,
                data = (int)(0.5 - (Input.GetAxisRaw("Vertical") / 2))//0=up 1=down
            };
            actionExecuted = true;
        }
        else if(Input.GetKeyDown(KeyCode.Space))
        {
            action = new Action
            {
                command = "Player action",
                id = 1,
                data = 4
            };
            actionExecuted = true;
        }

        if(actionExecuted)
        {
            string jsonString = JsonUtility.ToJson(action);
            System.IO.File.WriteAllText(filePath, jsonString);
            actionExecuted = false;
        }
        //string json = JsonConvert.SerializeObject(action.ToArray(), Formatting.Indented);
        //File.WriteAllText(@"/home/anavc/Toxic_Waste_Gym/Unity_Toxic Waste Gym/Assets/state.json", json);
    }
}
