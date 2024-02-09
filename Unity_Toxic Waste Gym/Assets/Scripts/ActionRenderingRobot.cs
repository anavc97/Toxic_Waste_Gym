using System.Collections;
using System.Collections.Generic;
using UnityEngine;

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
      while (gameData == null)
      { 
        Debug.Log("is gamedata null? " + gameData);
        gameHandler = GameObject.Find("GameHandler").GetComponent<GameHandler>();
      
        gameData = gameHandler.gameData;
        
      }
      BallsIdd = new Dictionary<string, bool>(); 
      Debug.Log("BallsIDD: " + BallsIdd);
    }

    // Update is called once per frame
    void Update()
    {

      Debug.Log("gamedata is not null: " + gameData.Data.Objects[1].Name);
      
      BallsIdd.Clear();

      foreach (var obj in gameData.Data.Objects)
      {   
        if (obj.HoldState == 0) 
        {
          BallsIdd[obj.Name] = false;

        }
          
      } 

      if (BallsIdd[gameData.Data.Objects[0].Name] == false)
      {
        moveOrRotate(new Vector3(gameData.Data.Objects[0].Position[1],14-gameData.Data.Objects[0].Position[0],0), new Vector2(0,1));
      }
      
    }

    public void moveOrRotate(Vector3 newPosition, Vector2 newOrientation)
    { 
      //Debug.Log(transform.position);
      //Debug.Log(newPosition); 
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


    public void interactWithBall(bool newBallState)
    {
      if(hasBall != newBallState){ //if robot starts holding a ball
        hasBall = newBallState;
        animator.SetBool("hasBall", hasBall);
      }
    }

}
