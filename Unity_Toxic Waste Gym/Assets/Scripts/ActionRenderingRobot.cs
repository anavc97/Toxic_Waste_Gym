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

    void Awake()
    {
      animator = GetComponent<Animator>();
    }

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
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
        GameObject ball = FindClosestBall();
        StartCoroutine(StartIdAnimation(ball));
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
      GameObject load = ball.transform.Find("Load").gameObject;
      GameObject text = ball.transform.Find("Text").gameObject;
      load.GetComponent<TextMeshPro>().enabled = true;
      Debug.Log("load: " + load.GetComponent<TextMeshPro>().enabled);
      yield return new WaitForSeconds(3f);
      load.GetComponent<TextMeshPro>().enabled = false;
      text.GetComponent<TextMeshPro>().enabled = true;
    }

    public GameObject FindClosestBall()
    {
      GameObject[] balls = GameObject.FindGameObjectsWithTag("Ball");

      // Initialize variables to keep track of the closest ball and its distance
      GameObject closestBall = null;
      float closestDistance = Mathf.Infinity;

      // Find the closest ball
      foreach (GameObject ball in balls)
      {
          float distance = Vector3.Distance(transform.position, ball.transform.position);
          if (distance < closestDistance)
          {
              closestDistance = distance;
              closestBall = ball;
          }
      }

      return closestBall;
    }

    public void interactWithBall(bool newBallState)
    {
      if(hasBall != newBallState){ //if robot starts holding a ball
        hasBall = newBallState;
        animator.SetBool("hasBall", hasBall);
      }
    }

}
