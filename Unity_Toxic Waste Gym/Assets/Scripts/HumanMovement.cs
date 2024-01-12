using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Human_Movement : MonoBehaviour
{
    public float movementSpeed;
    public Transform movePoint;
    public LayerMask walls;

    private bool waiting = false;
    private Animator animator;
    /*[SerializeField] float movementSpeed;
    float vertical, horizontal;
    Rigidbody2D myRigidbody2D;*/

    void Awake()
    {
      animator = GetComponent<Animator>();
    }

    // Start is called before the first frame update
    void Start()
    {
      movePoint.parent = null;
      //myRigidbody2D = GetComponent<Rigidbody2D>();  
    }

    // Update is called once per frame
    void Update()
    {
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

        /*horizontal = Input.GetAxis("Horizontal");
        vertical = Input.GetAxis("Vertical");
        if (horizontal != 0) vertical = 0;
        myRigidbody2D.velocity = new Vector2(horizontal * movementSpeed, vertical * movementSpeed);*/
    }

    private bool Rotate(string direction, string animationParameter, string oppositeAnimationParameter)
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
    }
    
    private IEnumerator Wait()
    {
      waiting = true;
      yield return new WaitForSeconds(0.2f);
      waiting = false;
    }

}
