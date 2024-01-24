using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ActionRenderingRobot : MonoBehaviour
{
    public float movementSpeed;

    private bool hasBall = false;
    private Animator animator;

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
     
    }

    public void moveOrRotate(Vector3 newPosition, Vector2 newOrientation)
    { 
      Debug.Log(transform.position);
      Debug.Log(newPosition); 
      Debug.Log(newOrientation);

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
