using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Human_Movement : MonoBehaviour
{
    public float movementSpeed;
    public Transform movePoint;
    /*[SerializeField] float movementSpeed;
    float vertical, horizontal;
    Rigidbody2D myRigidbody2D;*/

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

      if(Vector3.Distance(transform.position, movePoint.position) <= .05f)
      {
        if(Mathf.Abs(Input.GetAxisRaw("Horizontal")) == 1f)
        {
          movePoint.position += new Vector3(Input.GetAxisRaw("Horizontal"), 0f, 0f);
        }
        else if(Mathf.Abs(Input.GetAxisRaw("Vertical")) == 1f)
        {
          movePoint.position += new Vector3(0f, Input.GetAxisRaw("Vertical"), 0f);
        }
         
      }
        /*horizontal = Input.GetAxis("Horizontal");
        vertical = Input.GetAxis("Vertical");
        if (horizontal != 0) vertical = 0;
        myRigidbody2D.velocity = new Vector2(horizontal * movementSpeed, vertical * movementSpeed);*/
    }
}
