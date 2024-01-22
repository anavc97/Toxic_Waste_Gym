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
        public Vector2 Move; //Movement position 
        public int Grab; //1 if player has clicked space, 0 otherwise
    }
    
    private string filePath =  Application.dataPath + "/human_action.json";
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {       
        if(Mathf.Abs(Input.GetAxisRaw("Horizontal")) == 1f)
        {
            Action action = new Action
            {
                Move = new Vector2(Input.GetAxisRaw("Horizontal"),0),
                Grab = 0
            };
            string jsonString = JsonUtility.ToJson(action);
            System.IO.File.WriteAllText(filePath, jsonString);
        }
        else if(Mathf.Abs(Input.GetAxisRaw("Vertical")) == 1f)
        {
            Action action = new Action
            {
                Move = new Vector2(0,Input.GetAxisRaw("Vertical")),
                Grab = 0
            };
            string jsonString = JsonUtility.ToJson(action);
            System.IO.File.WriteAllText(filePath, jsonString);
        }
        else if(Input.GetKeyDown(KeyCode.Space))
        {
            Action action = new Action
            {
                Move = new Vector2(0,0),
                Grab = 1
            };
            string jsonString = JsonUtility.ToJson(action);
            System.IO.File.WriteAllText(filePath, jsonString);
        }
        //string json = JsonConvert.SerializeObject(action.ToArray(), Formatting.Indented);
        //File.WriteAllText(@"/home/anavc/Toxic_Waste_Gym/Unity_Toxic Waste Gym/Assets/state.json", json);
    }
}
