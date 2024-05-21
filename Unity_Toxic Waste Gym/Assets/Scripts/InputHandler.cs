using System.Collections;
using UnityEngine;
using Newtonsoft.Json;


public class InputHandler : MonoBehaviour
{
    [System.Serializable]
    public class Action
    {
        public string command;
        public ActionData data; 
    }
    public class ActionData
    {
        public int id; // human = 0 astro = 1
        public int action; //0=up, 1=down, 2=left, 3=right, 4=interact
    }
    private string filePath =  Application.dataPath + "/human_action.json";
    private bool actionExecuted = false;
    public bool sendAction = true;
    private bool waiting = false;
    private bool gameOver;
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {   
        gameOver = GameObject.Find("GameHandler").GetComponent<GameHandler>().gameOver;
        Action action = new Action();
        action.command = "p_act";
        action.data = new ActionData();
        
        if (sendAction && !gameOver)
        {
            if(Input.GetKeyDown(KeyCode.RightArrow) || Input.GetKeyDown(KeyCode.LeftArrow))
            {   
                action.data.id = 0;
                action.data.action = (int)(2.5 + (Input.GetAxisRaw("Horizontal") / 2)); //2=left 3=right
                actionExecuted = true;
                if (!waiting)
                {
                    StartCoroutine(Wait());
                }
            }
            else if(Input.GetKeyDown(KeyCode.UpArrow) || Input.GetKeyDown(KeyCode.DownArrow))
            {   
                action.data.id = 0;
                action.data.action = (int)(0.5 - (Input.GetAxisRaw("Vertical") / 2));//0=up 1=down
                actionExecuted = true;
                if (!waiting)
                {
                    StartCoroutine(Wait());
                }
            }
            else if(Input.GetKeyDown(KeyCode.Space))
            {   
                action.data.id = 0;
                action.data.action = 4;
                actionExecuted = true;
                if (!waiting)
                {
                    StartCoroutine(Wait());
                }
            }

            if(actionExecuted)
            {   
                actionExecuted = false;
                string jsonString = JsonConvert.SerializeObject(action);
                sendAction = false;
                GameObject.Find("GameHandler").GetComponent<GameHandler>().SendActionMessage(jsonString);
            }
        }
 
        //string json = JsonConvert.SerializeObject(action.ToArray(), Formatting.Indented);
        //File.WriteAllText(@"/home/anavc/Toxic_Waste_Gym/Unity_Toxic Waste Gym/Assets/state.json", json);
    }

    private IEnumerator Wait()
    {
      waiting = true;
      yield return new WaitForSeconds(0.2f);
      waiting = false;
    }

}
