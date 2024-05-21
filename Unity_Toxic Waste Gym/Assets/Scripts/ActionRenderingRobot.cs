using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using Newtonsoft.Json;

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

    private Animator animator;
    public Dictionary<string, bool> BallsIdd;

    public GameHandler gameHandler;
    public BallInteraction ballInteraction;
    //public GameObject chatBot;
    public GameObject historyChat;
    public GameObject currentChat;
    public GameObject humanPlayer;
    private GameObject bubble;
    private GameObject load;
    private GameObject text;
    private List<Vector3> gridPositions; 

    private bool identifying = false;
    private int currentChatNumber = 1;

    public List<Vector3> pos_around = new List<Vector3>();
    public Dictionary<int, int> oppositeStepActions = new Dictionary<int, int>(){{0,1},{1,0},{2,3},{3,2}};
    public float pos_x;
    public float pos_y;
    public string jsonString;
    public int previousStepAction = 0;
    public Vector3 astroStation;
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
      //Debug.Log("Start of action rendering robot evoked");
      action.command = "p_act";
      historyChat = GameObject.Find("HistoryChat"); //Migrated to BallInteraction
      humanPlayer = GameObject.Find("human");
      ballInteraction = GameObject.Find("red_1").GetComponent<BallInteraction>();
      Scene currentScene = SceneManager.GetActiveScene();
		  if(currentScene.name == "level_one"){defineGridLevelOne(); astroStation = new Vector3(6, 6,0);}
      else if(currentScene.name == "level_two"){defineGridLevelTwo(); astroStation = new Vector3(7, 7,0);}
      StartCoroutine(AstroAutomatic());
      //Debug.Log("Start of action rendering robot executed");

    }

    void defineGridLevelOne()
    {
     gridPositions = new List<Vector3>();

      // Add positions to the list using nested loops
      for (int x = 0; x <= 14; x++)
      {
          for (int y = 0; y <= 14; y++)
          {
              // Add positions according to the specified patterns
              if ((y == 0 && (x == 0 || x == 14)) || (y == 14 && (x == 0 || x == 14))) //Corners
                  gridPositions.Add(new Vector3(x, y,0));
              else if ((x == 0 && (y >= 1 && y < 14)) || (x == 14 && (y >= 1 && y < 14))) //Outside horizontal
                  gridPositions.Add(new Vector3(x, y,0));
              else if ((y == 0 && (x >= 1 && x < 14)) || (y == 0 && (x >= 1 && x < 14)))//Outside vertical
                  gridPositions.Add(new Vector3(x, y,0));
              else if ((x == 1 && y == 2) || (x == 1 && y == 3) || (x == 2 && (y == 2 || y == 3)) ||
                        ((x >= 5 && x <= 9) && y == 2) || ((x >= 12 && x <= 13) && y == 2) ||
                        ((x >= 6 && x <= 8) && y == 3) || ((x >= 6 && x <= 8) && y == 4) || ((x >= 10 && x <= 11) && y == 4) ||
                        (x == 7 && (y == 5 || y == 6 || y == 7)) || ((x >= 10 && x <= 11) && (y == 5 || y == 6 || y == 7)) ||
                        (x == 4 && y == 5) || ((x >= 2 && x <= 4) && y == 6) || ((x == 9 || x == 12) && y == 6) || 
                        (x == 5 && y == 8) || ((x >= 10 && x <= 11) && y == 8) || ((x >= 1 && x <= 5) && y == 9) ||
                        (x == 7 && y == 9) || ((x >= 1 && x <= 5) && y == 10) || ((x >= 8 && x <= 11) && y == 11) ||
                        ((x >= 2 && x <= 6) && y == 12) || ((x >= 10 && x <= 11) && y == 12))
                  gridPositions.Add(new Vector3(x, y,0));
          }
      }
    }
    //Change to be accurate with level two positions
    void defineGridLevelTwo()
    {
     gridPositions = new List<Vector3>();

      // Add positions to the list using nested loops
      for (int x = 0; x <= 14; x++)
      {
          for (int y = 0; y <= 14; y++)
          {
              // Add positions according to the specified patterns
              if ((y == 0 && (x == 0 || x == 14)) || (y == 14 && (x == 0 || x == 14))) //Corners
                  gridPositions.Add(new Vector3(x, y,0));
              else if ((x == 0 && (y >= 1 && y < 14)) || (x == 14 && (y >= 1 && y < 14))) //Outside horizontal
                  gridPositions.Add(new Vector3(x, y,0));
              else if ((y == 0 && (x >= 1 && x < 14)) || (y == 0 && (x >= 1 && x < 14)))//Outside vertical
                  gridPositions.Add(new Vector3(x, y,0));
              else if ((y == 1 && (x == 1 || x == 2)) || (y == 1 && (x >= 7 && x <= 10)) || 
                        (y == 1 && (x == 12 || x == 13)) || (y == 2 && (x == 9 || x == 10)) ||
                        (y == 2 && (x == 12 || x == 13)) || (y == 3 && (x == 9 || x == 10)) ||
                        (y == 3 && (x == 4 || x == 5)) || (y == 4 && (x >= 2 && x <= 7)) ||
                        (y == 5 && (x >= 2 && x <= 5)) || (y == 5 && (x >= 9 && x <= 11)) ||
                        (y == 6 && (x >= 2 && x <= 5)) || (y == 6 && (x >= 7 && x <= 11)) ||
                        (y == 7 && (x == 2 || x == 3)) || (y == 8 && x == 2) || (y == 9 && x == 4) ||
                        (y == 9 && (x >= 6 && x <= 11)) || (y == 10 && (x >= 6 && x <= 11)) ||
                        (y == 10 && (x >= 1 && x <= 3)) || (x == 6 && (y >= 11 && y <= 13)) ||
                        (y == 11 && (x == 10 || x == 11)) || (y == 12 && (x == 10 || x == 11)) ||
                        (y == 12 && (x >= 2 && x <= 5)) || (x == 8 && (y == 12 && y == 13)))
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
        //StartCoroutine(StartIdAnimation(ball));
      }
    }


    IEnumerator AstroAutomatic()
    { 
      yield return new WaitForSeconds(1f);
      GameObject[] balls = GameObject.FindGameObjectsWithTag("Ball");
      GameObject[] allBalls = GameObject.FindGameObjectsWithTag("Ball");
      float distanceToHuman = Mathf.Infinity;
      //Debug.Log("Started astro automatic");

      while (balls.Length != 0)
      { 
        float closestDistance = Mathf.Infinity;
        balls = GameObject.FindGameObjectsWithTag("Ball");
        if(balls.Length == 0){break;}
        GameObject randomBall = balls[Random.Range(0, balls.Length)];

        GameObject[] identifiedBalls = GameObject.FindGameObjectsWithTag("IDdBall");
        allBalls = new GameObject[balls.Length + identifiedBalls.Length];
        balls.CopyTo(allBalls, 0);
        identifiedBalls.CopyTo(allBalls, balls.Length);

        while (closestDistance >  Mathf.Sqrt(2))
        { 
          //Move towards human if human is holding a ball
          while(humanHoldingBall(allBalls))
          {
            if(distanceToHuman <= Mathf.Sqrt(2))
            {
              yield return new WaitForSeconds(0.4f);
              distanceToHuman = Vector3.Distance(transform.position, humanPlayer.transform.position);
              continue;
            }
            distanceToHuman = Vector3.Distance(transform.position, humanPlayer.transform.position);
            ObtainNextAction(humanPlayer.transform.position);
            yield return new WaitForSeconds(0.4f);
          }
          
          //Move towards closest ball if human isn't holding a ball
          balls = GameObject.FindGameObjectsWithTag("Ball");
          if(balls.Length ==0){break;}
          if(!arrayContains(balls,randomBall)){randomBall = balls[Random.Range(0, balls.Length)];}
          //Debug.Log("Entered while to move towards nearest ball");
          /*//Go to closer ball
          var results = FindClosestBall(balls);
          GameObject ball = results.Item1;
          closestDistance = results.Item2;*/
          //Go to random ball
          closestDistance = Vector3.Distance(transform.position, randomBall.transform.position);


          //ObtainNextAction(ball.transform.position);
          ObtainNextAction(randomBall.transform.position);
          //moveOrRotateRobot(next_step, new Vector2(0,-1));
          yield return new WaitForSeconds(0.4f);
        }
        

        //Identify ball (no need to send this action to backend since it doesn't affect game)
        balls = GameObject.FindGameObjectsWithTag("Ball");
        if(balls.Length == 0){break;}
        var resultsClosestBall = FindClosestBall(balls);
        GameObject closestBall = resultsClosestBall.Item1;
        StartCoroutine(ballInteraction.StartIdAnimation(closestBall));
        
        /*action.data = new ActionDataRobot();
        action.data.id = 1;
        action.data.action = 4; 
        jsonString = JsonConvert.SerializeObject(action);
        Debug.Log("robot command sent: " + jsonString);
        GameObject.Find("GameHandler").GetComponent<GameHandler>().SendActionMessage(jsonString);*/
        yield return new WaitForSeconds(2.5f);
      }

      //When all balls identified robot still follows human when he's holding a ball
      //TODO: Add check to see if human next to robot without ball and if yes move away to not block him
      balls = GameObject.FindGameObjectsWithTag("IDdBall");
      while(balls.Length != 0)
      {
        distanceToHuman = Vector3.Distance(transform.position, humanPlayer.transform.position);
        if(humanHoldingBall(balls) && distanceToHuman > Mathf.Sqrt(2)) //Move towards human
        {
          ObtainNextAction(humanPlayer.transform.position);
        }
        else if(!humanHoldingBall(balls) && transform.position != astroStation) //Move towards station
        {
          ObtainNextAction(astroStation);
        }
        yield return new WaitForSeconds(0.4f);
        balls = GameObject.FindGameObjectsWithTag("IDdBall");
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
    /*IEnumerator StartIdAnimation(GameObject ball)
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
    }*/

    //Obtain next step action according to target destination (move towards human or ball)
    public void ObtainNextAction(Vector3 targetPosition)
    {
      pos_x = transform.position.x;
      pos_y = transform.position.y;
      pos_around = new List<Vector3>();
      pos_around.Add(new Vector3(pos_x, pos_y+1f,0)); // square up
      pos_around.Add(new Vector3(pos_x, pos_y-1f,0)); // square down
      pos_around.Add(new Vector3(pos_x-1f, pos_y,0)); // square left
      pos_around.Add(new Vector3(pos_x+1f, pos_y,0)); // square right
      int next_step = FindNextStep(pos_around, targetPosition);
      SendNextAction(next_step);
    }

    //Send next step action to perform to backend
    public void SendNextAction(int action_number)
    {
      action.data = new ActionDataRobot(); 
      action.data.id = 1;
      action.data.action = action_number;
      jsonString = JsonConvert.SerializeObject(action);
      //Debug.Log("robot command sent: " + jsonString);
      GameObject.Find("GameHandler").GetComponent<GameHandler>().SendActionMessage(jsonString);
    }


    public (GameObject,float) FindClosestBall(GameObject[] balls)
    {

      // Initialize variables to keep track of the closest ball and its distance
      GameObject closestBall = null;
      float closestDistance = Mathf.Infinity;

      // Find the closest ball
      foreach (GameObject ball in balls)
      {   
        if (ball != null && ball.GetComponent<SpriteRenderer>().enabled)
        {
          float distance = Vector3.Distance(transform.position, ball.transform.position);
          if (distance < closestDistance)
          {
              closestDistance = distance;
              closestBall = ball;
          }
        }
      }
      //Debug.Log("Ball: " + closestBall.name + " pos: " + closestBall.transform.position);
      return (closestBall,closestDistance);
    }

    
    //public Vector3 FindNextStep(List<Vector3> stepList, GameObject ball)
    public int FindNextStep(List<Vector3> stepList, Vector3 distantObjectPosition)
    {
      // Initialize variables to keep track of the closest ball and its distance
      //Dictionary<Vector3, float> Steps = new Dictionary<Vector3, float>();
      Dictionary<int, float> Steps = new Dictionary<int, float>();
      //List<Vector3> closestSteps = new List<Vector3>();
      List<int> closestStepsAction = new List<int>();
      float closestDistance = Mathf.Infinity;
      int previousStepActionOpposite = oppositeStepActions[previousStepAction];
      //Debug.Log("Current Position X: " + transform.position.x + " Y: " + transform.position.y);

      // Find the closest tile to distantObject
      foreach (Vector3 step in stepList)
      {   
        if (!gridPositions.Contains(step))
        {
          float distance = Vector3.Distance(step, distantObjectPosition);
          Steps.Add(stepList.IndexOf(step), distance);
          //Debug.Log("New possible step action added: " + stepList.IndexOf(step) + " with distance: " + distance);
          if (distance < closestDistance)
          {
              closestDistance = distance;
          }
        }
      }
      //List<Vector3> keyList = new List<Vector3>(Steps.Keys);
      /*List<int> keyList = new List<int>(Steps.Keys);
      if (Random.value <0.1f && !identifying)
      { 
        int i = Random.Range(0, keyList.Count);
        previousStepAction = keyList[i];
        Debug.Log("Random action selected: " + keyList[i]);
        return keyList[i];
      }*/
          //return keyList[i];}

      foreach (var c_step in Steps)
      {
        if(c_step.Value == closestDistance)
        {
          //closestSteps.Add(c_step.Key);
          closestStepsAction.Add(c_step.Key);
        }
      }

      //Prevent robot from getting stuck in a cycle moving back and forth to same positions
      if(Steps.Count > 1 && closestStepsAction.Contains(previousStepActionOpposite))
      {
        //Debug.Log("Previous step action opposite: " + previousStepActionOpposite);
        //Debug.Log("All possible next steps: " + closestStepsAction.Count);
        closestStepsAction.Remove(previousStepActionOpposite);
        if(closestStepsAction.Count == 0)
        {
          closestDistance = Mathf.Infinity;
          closestStepsAction = new List<int>();
          foreach (var step in Steps)
          {
            if(step.Value <= closestDistance && step.Key != previousStepActionOpposite)
            {
              closestStepsAction.Add(step.Key);
              closestDistance = step.Value;
            }
          }
          //Debug.Log("Next necessary steps: " + closestStepsAction.Count);
        }
      }
      previousStepAction = closestStepsAction[Random.Range(0, closestStepsAction.Count)];
      return previousStepAction;

      //return closestSteps[Random.Range(0, closestSteps.Count)];
      //return closestStepsAction[Random.Range(0, closestStepsAction.Count)];
    }

    //Check if human is currently holding a ball
    public bool humanHoldingBall(GameObject[] balls)
    {
      foreach (GameObject ball in balls)
      {   
        if(ball != null && !ball.GetComponent<SpriteRenderer>().enabled)
        {
          return true;
        }
      }
      return false;
    }

    public bool arrayContains (GameObject[] array, GameObject objToCheck) 
    {
      foreach (GameObject obj in array) {if(obj == objToCheck) return true;}
      return false;
    }
}
