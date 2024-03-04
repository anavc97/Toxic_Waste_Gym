using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using System.Threading;
using Newtonsoft.Json;
using System.Net;
using System.Net.Sockets;
using System.Text;

public class ActionRenderingRobot : MonoBehaviour
{
    [System.Serializable]
    public class ActionRobot
    {
        public string command;
        public ActionDataRobot data; 
    }
    public class ActionDataRobot
    {
        public int id; // astro = 1
        public int action; //0=up, 1=down, 2=left, 3=right, 4=interact
    }
    
    public float movementSpeed;
    public ActionRobot action = new ActionRobot();

    private bool hasBall = false;
    private Animator animator;
    public Dictionary<string, bool> BallsIdd;

    public GameHandler gameHandler;
    //public GameObject chatBot;
    public GameObject historyChat;
    public GameObject currentChat;
    private GameObject bubble;
    private GameObject load;
    private GameObject text;
    private List<Vector3> gridPositions; 

    private bool identifying = false;
    private int currentChatNumber = 1;

    public List<Vector3> pos_around = new List<Vector3>();
    public float pos_x;
    public float pos_y;
    public string jsonString;

    void Awake()
    {
      animator = GetComponent<Animator>();
    }

    // Start is called before the first frame update
    void Start()
    {
      //chatBot = GameObject.Find("ChatBot");
      //bubble = chatBot.transform.Find("Bubble").gameObject;
      //load = bubble.transform.Find("Load").gameObject;
      //text = bubble.transform.Find("Text").gameObject;
      Debug.Log("Start of action rendering robot evoked");
      action.command = "p_act";
      historyChat = GameObject.Find("HistoryChat"); //Migrated to BallInteraction
      defineGrid();
      StartCoroutine(AstroAutomatic());
      Debug.Log("Start of action rendering robot executed");

    }

    // Not used
    void defineGrid()
    {
     gridPositions = new List<Vector3>();

      // Add positions to the list using nested loops
      for (int x = 0; x <= 14; x++)
      {
          for (int y = 0; y <= 14; y++)
          {
              // Add positions according to the specified patterns
              if ((y == 0 && (x == 0 || x == 14)) || (y == 14 && (x == 0 || x == 14)))
                  gridPositions.Add(new Vector3(x, y,0));
              else if ((x == 0 && (y >= 1 && y <= 14)) || (x == 14 && (y >= 1 && y <= 14)))
                  gridPositions.Add(new Vector3(x, y,0));
              else if ((x == 1 && y == 2) || (x == 2 && (y == 2 || y == 3)) || ((x >= 5 && x <= 9) && y == 2) ||
                        ((x >= 12 && x <= 14) && y == 2) || (x == 1 && y == 3) || (x == 2 && y == 3) ||
                        ((x >= 6 && x <= 8) && y == 3) || ((x >= 6 && x <= 8) && y == 4) || ((x >= 10 && x <= 11) && y == 4) ||
                        (x == 7 && (y == 5 || y == 6 || y == 7)) || ((x >= 10 && x <= 11) && (y == 5 || y == 6 || y == 7)) ||
                        ((x >= 4 && x <= 5) && y == 5) || ((x >= 2 && x <= 4) && y == 6) || (x == 7 && y == 6) ||
                        ((x >= 9 && x <= 12) && y == 6) || (x == 7 && y == 7) || ((x >= 10 && x <= 11) && y == 7) ||
                        (x == 5 && y == 8) || ((x >= 10 && x <= 11) && y == 8) || ((x >= 1 && x <= 5) && y == 9) ||
                        (x == 7 && y == 9) || ((x >= 1 && x <= 5) && y == 10) || ((x >= 8 && x <= 11) && y == 11) ||
                        ((x >= 2 && x <= 6) && y == 12) || ((x >= 10 && x <= 11) && y == 12))
                  gridPositions.Add(new Vector3(x, y,0));
          }
      }
    }

    void Update()
    {
      
    }
    
    //Not used
    public void AstroManual()
    {
      float pos_x = transform.position.x;
      float pos_y = transform.position.y;

      if (Input.GetKeyDown(KeyCode.A))//move left
      {
        moveOrRotateRobot(new Vector3(pos_x-1,pos_y,0), new Vector2(-1,0));
      }
      else if(Input.GetKeyDown(KeyCode.D))// move right
      {   
        moveOrRotateRobot(new Vector3(pos_x+1,pos_y,0), new Vector2(1,0));
      }
      else if(Input.GetKeyDown(KeyCode.W))// up
      {   
        moveOrRotateRobot(new Vector3(pos_x,pos_y+1,0), new Vector2(0,1));
      }
      else if(Input.GetKeyDown(KeyCode.S)) //down
      {   
        moveOrRotateRobot(new Vector3(pos_x,pos_y-1,0), new Vector2(0,-1));
      }
      else if(Input.GetKeyDown(KeyCode.E))
      { 
        GameObject[] balls = GameObject.FindGameObjectsWithTag("Ball");  
        var results = FindClosestBall(balls);
        GameObject ball = results.Item1;
        StartCoroutine(StartIdAnimation(ball));
      }
    }

    
    IEnumerator AstroAutomatic()
    { 
      GameObject[] balls = GameObject.FindGameObjectsWithTag("Ball");
      GameObject[] allBalls = GameObject.FindGameObjectsWithTag("Ball");
      Debug.Log("Started astro automatic");

      //Missing check to see if human is holding a ball (if yes move towards him)
      while (balls != null)
      { 
        float closestDistance = Mathf.Infinity;
        float distanceToHuman = Mathf.Infinity;
        balls = GameObject.FindGameObjectsWithTag("Ball");

        GameObject[] identifiedBalls = GameObject.FindGameObjectsWithTag("IDdBall");
        allBalls = new GameObject[balls.Length + identifiedBalls.Length];
        balls.CopyTo(allBalls, 0);
        identifiedBalls.CopyTo(allBalls, balls.Length);

        while (closestDistance >  Mathf.Sqrt(2))
        { 
          //Move towards human if human is holding a ball
          while(humanHoldingBall(allBalls) && distanceToHuman > Mathf.Sqrt(2))
          {
            pos_x = transform.position.x;
            pos_y = transform.position.y;
            pos_around = new List<Vector3>();
            pos_around.Add(new Vector3(pos_x, pos_y+1f,0)); // square up
            pos_around.Add(new Vector3(pos_x, pos_y-1f,0)); // square down
            pos_around.Add(new Vector3(pos_x-1f, pos_y,0)); // square left
            pos_around.Add(new Vector3(pos_x+1f, pos_y,0)); // square right
            GameObject humanPlayer = GameObject.Find("human");
            int next_step_human = FindNextStep(pos_around, humanPlayer);
            distanceToHuman = Vector3.Distance(transform.position, humanPlayer.transform.position);

            action.data = new ActionDataRobot(); 
            action.data.id = 1;
            action.data.action = next_step_human;
            jsonString = JsonConvert.SerializeObject(action);
            Debug.Log("robot command sent: " + jsonString);
            GameObject.Find("GameHandler").GetComponent<GameHandler>().SendActionMessage(jsonString);
            yield return new WaitForSeconds(0.5f);
          }
          
          //Move towards closest ball if human isn't holding a ball
          Debug.Log("Entered while to move towards nearest ball");
          var results = FindClosestBall(balls);
          GameObject ball = results.Item1;
          closestDistance = results.Item2;

          pos_x = transform.position.x;
          pos_y = transform.position.y;
          pos_around = new List<Vector3>();
          pos_around.Add(new Vector3(pos_x, pos_y+1f,0)); // square up
          pos_around.Add(new Vector3(pos_x, pos_y-1f,0)); // square down
          pos_around.Add(new Vector3(pos_x-1f, pos_y,0)); // square left
          pos_around.Add(new Vector3(pos_x+1f, pos_y,0)); // square right


          //Vector3 next_step = FindNextStep(pos_around, ball);
          int next_step = FindNextStep(pos_around, ball);
          action.data = new ActionDataRobot(); 
          action.data.id = 1;
          action.data.action = next_step;
          jsonString = JsonConvert.SerializeObject(action);
          Debug.Log("robot command sent: " + jsonString);
          GameObject.Find("GameHandler").GetComponent<GameHandler>().SendActionMessage(jsonString);

          //moveOrRotateRobot(next_step, new Vector2(0,-1));
          
          yield return new WaitForSeconds(0.5f);
        }
        
        //var res = FindClosestBall(balls);
        //GameObject IDball = res.Item1;
        //StartCoroutine(StartIdAnimation(IDball));

        //Identify ball
        action.data = new ActionDataRobot();
        action.data.id = 1;
        action.data.action = 4; 
        jsonString = JsonConvert.SerializeObject(action);
        Debug.Log("robot command sent: " + jsonString);
        GameObject.Find("GameHandler").GetComponent<GameHandler>().SendActionMessage(jsonString);
        yield return new WaitForSeconds(3f);
      }
    }


    public void moveOrRotateRobot(Vector3 newPosition, Vector2 newOrientation)
    { 
      if(animator.GetFloat("moveX")!=newOrientation.x || animator.GetFloat("moveY")!=newOrientation.y)
      {
        animator.SetFloat("moveX", newOrientation.x);
        animator.SetFloat("moveY", newOrientation.y);
      }
      else if(transform.position != newPosition)
      {
        Vector3 human_position = GameObject.Find("human").transform.position;
        if(human_position.x != newPosition.x || human_position.y != newPosition.y)
        {
          transform.position = Vector3.MoveTowards(transform.position, newPosition, movementSpeed * Time.deltaTime);
        }    
      }
    }
  
    //Not used - Migrated to BallInteraction script
    IEnumerator StartIdAnimation(GameObject ball)
    {
      
      identifying = true;
      string type = ball.name.Split('_')[0];
      Debug.Log("Ball ID: " + type);
      
      string currentChatName = "Chat" + currentChatNumber;
      currentChat = historyChat.transform.Find(currentChatName).gameObject;
      currentChat.SetActive(true);
      bubble = currentChat.transform.Find("Bubble").gameObject;
      load = bubble.transform.Find("Load").gameObject;
      text = bubble.transform.Find("Text").gameObject;

      if (text.GetComponent<TextMeshPro>().enabled)
      {
        text.GetComponent<TextMeshPro>().enabled = false;
      }
      load.GetComponent<TextMeshPro>().enabled = true;
      ball.tag = "IDdBall";
      yield return new WaitForSeconds(3f);
      load.GetComponent<TextMeshPro>().enabled = false;
      text.GetComponent<TextMeshPro>().enabled = true;
      text.GetComponent<TextMeshPro>().text = $"This is a {type} ball!";
      currentChatNumber += 1;
      identifying = false;
    }

    public (GameObject,float) FindClosestBall(GameObject[] balls)
    {

      // Initialize variables to keep track of the closest ball and its distance
      GameObject closestBall = null;
      float closestDistance = Mathf.Infinity;

      // Find the closest ball
      foreach (GameObject ball in balls)
      {   
        if (ball.GetComponent<SpriteRenderer>().enabled)
        {
          float distance = Vector3.Distance(transform.position, ball.transform.position);
          if (distance < closestDistance)
          {
              closestDistance = distance;
              closestBall = ball;
          }
        }
      }
      Debug.Log("Ball: " + closestBall.name + " pos: " + closestBall.transform.position);
      return (closestBall,closestDistance);
    }

    
    //public Vector3 FindNextStep(List<Vector3> stepList, GameObject ball)
    public int FindNextStep(List<Vector3> stepList, GameObject distantObject)
    {
      // Initialize variables to keep track of the closest ball and its distance
      //Dictionary<Vector3, float> Steps = new Dictionary<Vector3, float>();
      Dictionary<int, float> Steps = new Dictionary<int, float>();
      //List<Vector3> closestSteps = new List<Vector3>();
      List<int> closestStepsAction = new List<int>();
      float closestDistance = Mathf.Infinity;

      // Find the closest tile to distantObject
      foreach (Vector3 step in stepList)
      {   
        if (!gridPositions.Contains(step))
        {
          float distance = Vector3.Distance(step, distantObject.transform.position);
          Steps.Add(stepList.IndexOf(step), distance);
          if (distance < closestDistance)
          {
              closestDistance = distance;
          }
        }
      }
      //List<Vector3> keyList = new List<Vector3>(Steps.Keys);
      List<int> keyList = new List<int>(Steps.Keys);
      if (Random.value <0.1f && !identifying)
        { int i = Random.Range(0, keyList.Count);
        return keyList[i];}
          //return keyList[i];}

      foreach (var c_step in Steps)
      {
        if(c_step.Value == closestDistance)
        {
          //closestSteps.Add(c_step.Key);
          closestStepsAction.Add(c_step.Key);
        }
      }

      //return closestSteps[Random.Range(0, closestSteps.Count)];
      return closestStepsAction[Random.Range(0, closestStepsAction.Count)];
    }

    public bool humanHoldingBall(GameObject[] balls)
    {
      foreach (GameObject ball in balls)
      {   
        if(!ball.GetComponent<SpriteRenderer>().enabled)
        {
          return true;
        }
      }
      return false;
    }

}
