using System.Collections;
using System.Collections.Generic;
using UnityEngine;
//using System.Text.Json;
//using System.Text.Json.Serialization;
//using Newtonsoft.Json;


/*public class Action
{
    public Vector2 move;
    public int Grab;
}*/


public class NewBehaviourScript : MonoBehaviour
{
    //private List<Action> action = new List<Action>();
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
       /* if(Mathf.Abs(Input.GetAxisRaw("Horizontal")) == 1f)
        {
            action.Add(new Action(new Vector2(Input.GetAxisRaw("Horizontal"),0),0));
        }
        else if(Mathf.Abs(Input.GetAxisRaw("Vertical")) == 1f)
        {
            action.Add(new Action(new Vector2(0,Input.GetAxisRaw("Vertical")),0));
        }
        else if(Input.GetKeyDown(KeyCode.Space))
        {
            action.Add(new Action(new Vector2(0,0),1));
        }
        string json = JsonConvert.SerializeObject(action.ToArray(), Formatting.Indented);
        File.WriteAllText(@"/home/anavc/Toxic_Waste_Gym/Unity_Toxic Waste Gym/Assets/state.json", json);*/
    }
}
