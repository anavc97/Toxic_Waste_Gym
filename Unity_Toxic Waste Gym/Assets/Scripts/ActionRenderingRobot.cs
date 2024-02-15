using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using System.Threading;

public class ActionRenderingRobot : MonoBehaviour
{
    public float movementSpeed;

    private bool hasBall = false;
    private Animator animator;
    public Dictionary<string, bool> BallsIdd;

    public GameHandler gameHandler;
    public GameHandler.GameData gameData;
    public GameObject chatBot;
    private GameObject bubble;
    private GameObject load;
    private GameObject text;
    private List<Vector3> gridPositions; 

    private 

    void Awake()
    {
      animator = GetComponent<Animator>();
    }

    // Start is called before the first frame update
    void Start()
    {
      chatBot = GameObject.Find("ChatBot");
      bubble = chatBot.transform.Find("Bubble").gameObject;
      load = bubble.transform.Find("Load").gameObject;
      text = bubble.transform.Find("Text").gameObject;
      defineGrid();
      StartCoroutine(AstroAutomatic());

    }

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
              else if ((x == 0 && (y >= 1 && y <= 13)) || (x == 14 && (y >= 1 && y <= 13)))
                  gridPositions.Add(new Vector3(x, y,0));
              else if ((x == 1 && y == 2) || (x == 2 && (y == 2 || y == 3)) || ((x >= 5 && x <= 9) && y == 2) ||
                        ((x >= 12 && x <= 13) && y == 2) || (x == 1 && y == 3) || (x == 2 && y == 3) ||
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

      while (balls != null)
      { 
        float closestDistance = Mathf.Infinity;
        balls = GameObject.FindGameObjectsWithTag("Ball");

        while (closestDistance >  Mathf.Sqrt(2))
        { 
          var results = FindClosestBall(balls);
          GameObject ball = results.Item1;
          closestDistance = results.Item2;

          float pos_x = transform.position.x;
          float pos_y = transform.position.y;
          List<Vector3> pos_around = new List<Vector3>();
          pos_around.Add(new Vector3(pos_x, pos_y+1f,0)); // square up
          pos_around.Add(new Vector3(pos_x, pos_y-1f,0)); // square down
          pos_around.Add(new Vector3(pos_x-1f, pos_y,0)); // square left
          pos_around.Add(new Vector3(pos_x+1f, pos_y,0)); // square right


          Vector3 next_step = FindNextStep(pos_around, ball);
          moveOrRotateRobot(next_step, new Vector2(0,-1));
          
          yield return new WaitForSeconds(0.5f);
        }
        
        var res = FindClosestBall(balls);
        GameObject IDball = res.Item1;
        StartCoroutine(StartIdAnimation(IDball));
        yield return new WaitForSeconds(3f);
      }

      
    }
    public void moveOrRotateRobot(Vector3 newPosition, Vector2 newOrientation)
    { 
      //Debug.Log("Current position: " + transform.position);
      //Debug.Log("New position: " + newPosition); 
      //Debug.Log(newOrientation);

      if(transform.position != newPosition)
      {  
        transform.position = Vector3.MoveTowards(transform.position, newPosition, movementSpeed * Time.deltaTime);  
      }
      else if(animator.GetFloat("moveX")!=newOrientation.x || animator.GetFloat("moveY")!=newOrientation.y)
      {
        animator.SetFloat("moveX", newOrientation.x);
        animator.SetFloat("moveY", newOrientation.y);
      }
    }
  
    IEnumerator StartIdAnimation(GameObject ball)
    {
      Debug.Log("Ball ID: " + ball.name);
      if (text.GetComponent<TextMeshPro>().enabled)
      {
        text.GetComponent<TextMeshPro>().enabled = false;
      }
      //GameObject load = ball.transform.Find("Load").gameObject;
      //GameObject text = ball.transform.Find("Text").gameObject;
      load.GetComponent<TextMeshPro>().enabled = true;
      ball.tag = "IDdBall";
      yield return new WaitForSeconds(3f);
      load.GetComponent<TextMeshPro>().enabled = false;
      text.GetComponent<TextMeshPro>().enabled = true;
      text.GetComponent<TextMeshPro>().text = $"This is a {ball.name} ball!";
      yield return new WaitForSeconds(3f);
      text.GetComponent<TextMeshPro>().enabled = false;
      
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

    public Vector3 FindNextStep(List<Vector3> stepList, GameObject ball)
    {
      // Initialize variables to keep track of the closest ball and its distance
      Dictionary<Vector3, float> Steps = new Dictionary<Vector3, float>();
      List<Vector3> closestSteps = new List<Vector3>();
      float closestDistance = Mathf.Infinity;

      // Find the closest ball
      foreach (Vector3 step in stepList)
      {   
        if (!gridPositions.Contains(step))
        {
          float distance = Vector3.Distance(step, ball.transform.position);
          Steps.Add(step, distance);
          if (distance < closestDistance)
          {
              closestDistance = distance;
          }
        }
      }
      List<Vector3> keyList = new List<Vector3>(Steps.Keys);
      if (Random.value <0.1f)
        { int i = Random.Range(0, keyList.Count);
          return keyList[i];}

      foreach (var c_step in Steps)
      {
        if(c_step.Value == closestDistance)
        {
          closestSteps.Add(c_step.Key);
        }
      }

      return closestSteps[Random.Range(0, closestSteps.Count)];
    }

}
