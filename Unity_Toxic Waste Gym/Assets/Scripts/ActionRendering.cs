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
    }

    // Update is called once per frame
    void Update()
    {
      /*
      transform.position = Vector3.MoveTowards(transform.position, movePoint.position, movementSpeed * Time.deltaTime);
      
      //Move in a constant distance
      if(Vector3.Distance(transform.position, movePoint.position) <= .05f)
      {
        //If input is left or right arrows
        if(Mathf.Abs(Input.GetAxisRaw("Horizontal")) == 1f)
        { 

          //Check if human is already facing the correct way or needs to rotate first
          if(!Rotate("Horizontal", "moveX", "moveY") && !waiting)
          {
            //Is it trying to move into a wall?
            if(!Physics2D.OverlapCircle(movePoint.position + new Vector3(Input.GetAxisRaw("Horizontal"), 0f, 0f), .2f, walls))
            {
              movePoint.position += new Vector3(Input.GetAxisRaw("Horizontal"), 0f, 0f);
            }
          }
        }

        //If input is up or down arrows
        else if(Mathf.Abs(Input.GetAxisRaw("Vertical")) == 1f)
        {
          //Check if human is already facing the correct way or needs to rotate first
          if(!Rotate("Vertical", "moveY", "moveX") && !waiting)
          {
            //Is it trying to move into a wall?
            if(!Physics2D.OverlapCircle(movePoint.position + new Vector3(0f, Input.GetAxisRaw("Vertical"), 0f), .2f, walls))
            {
              movePoint.position += new Vector3(0f, Input.GetAxisRaw("Vertical"), 0f);
            }
          }
        }
      }
      if(Input.GetKeyDown(KeyCode.Space))
      {
        hasBall = !hasBall;
        animator.SetBool("hasBall", hasBall);
      }

        /*horizontal = Input.GetAxis("Horizontal");
        vertical = Input.GetAxis("Vertical");
        if (horizontal != 0) vertical = 0;
        myRigidbody2D.velocity = new Vector2(horizontal * movementSpeed, vertical * movementSpeed);*/
    }

    /*private bool Rotate(string direction, string animationParameter, string oppositeAnimationParameter)
    {
      if(animator.GetFloat(animationParameter) >= 0f && Input.GetAxisRaw(direction) == -1f)
      {
        animator.SetFloat(animationParameter, Input.GetAxisRaw(direction));
        animator.SetFloat(oppositeAnimationParameter, 0f);
        if (!waiting)
        {
          StartCoroutine(Wait());
        }
        return true;
      }
      else if(animator.GetFloat(animationParameter) <= 0f && Input.GetAxisRaw(direction) == 1f)
      {
        animator.SetFloat(animationParameter, Input.GetAxisRaw(direction));
        animator.SetFloat(oppositeAnimationParameter, 0f);
        if (!waiting)
        {
          StartCoroutine(Wait());
        }
        return true;
      }
      return false;
    }*/
    
    private IEnumerator Wait()
    {
      waiting = true;
      yield return new WaitForSeconds(0.2f);
      waiting = false;
    }

    public void moveOrRotate(Vector3 newPosition, Vector2 newOrientation)
    { 

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
