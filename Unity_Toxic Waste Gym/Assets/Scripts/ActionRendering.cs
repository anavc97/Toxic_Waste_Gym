using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ActionRendering : MonoBehaviour
{
    public float movementSpeed;
    public Transform movePoint;
    //public LayerMask walls;

    private bool waiting = false;
    private bool hasBall = false;
    private Animator animator;
    private List<Vector3> walls;
    private BallInteraction ballInteraction;
    private GameObject astroPlayer;
    private List<Vector3> doorPositions = new List<Vector3>();

    void Awake()
    {
      animator = GetComponent<Animator>();
    }

    // Start is called before the first frame update
    void Start()
    {
      movePoint.parent = null;
      movePoint.position = transform.position;
      walls = GameObject.Find("Grid").GetComponent<GridLimits>().gridPositions;
      ballInteraction = GameObject.Find("red_1").GetComponent<BallInteraction>();
      astroPlayer = GameObject.Find("astro");
      doorPositions = GameObject.Find("GameHandler").GetComponent<GameHandler>().doorPositions;
      this.GetComponent<SpriteRenderer>().enabled = true;
    }

    // Update is called once per frame
    void Update()
    {

    }

    private IEnumerator Wait()
    {
      waiting = true;
      yield return new WaitForSeconds(0.2f);
      waiting = false;
    }

    public void moveOrRotate(Vector3 newPosition, Vector2 newOrientation)
    { 
      foreach (Vector3 vector in doorPositions)
      {
          if (vector == newPosition)  
          {   
            this.GetComponent<SpriteRenderer>().enabled = false;
          }
      }

      if(animator.GetFloat("moveX")!=newOrientation.x || animator.GetFloat("moveY")!=newOrientation.y)
      {
        animator.SetFloat("moveX", newOrientation.x);
        animator.SetFloat("moveY", newOrientation.y);
      }
      if(transform.position != newPosition && !walls.Contains(newPosition)) //TODO: Fix human not moving with constant values (shorter steps)
      { //Also check if not moving into any ball or astro's position
        if(ballInteraction.checkPositionVacancy(newPosition) && astroPlayer.transform.position != newPosition)
        { 
          movePoint.position = newPosition;
          transform.position = Vector3.MoveTowards(transform.position, movePoint.position, movementSpeed * Time.deltaTime);  
          //transform.position = Vector3.MoveTowards(transform.position, newPosition, movementSpeed * Time.deltaTime);  
        }
      }
    }

    public void humanInteractWithBall()
    {
      //Human starts holding or drops a given ball
      hasBall = !hasBall;
      animator.SetBool("hasBall", hasBall);
    }

    public bool getHasBall(){return hasBall;}
}
