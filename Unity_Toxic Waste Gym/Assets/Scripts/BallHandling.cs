using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BallHandling : MonoBehaviour
{
    public Transform movePoint;
    public Transform rayPoint;
    public float rayDistance;

    private GameObject grabbedObject;
    private int layerIndex;
    // Start is called before the first frame update
    void Start()
    {
        layerIndex = LayerMask.NameToLayer("Balls");
    }

    // Update is called once per frame
    void Update()
    {
        // RaycastHit2D hitInfo = Physics2D.Raycast(rayPoint.position, transform.right, rayDistance);

        if(Input.GetKeyDown(KeyCode.Space))
        {
            Debug.Log("Space Pressed");   
            foreach (GameObject ball in GameObject.FindGameObjectsWithTag("Ball"))
            {
                if(Mathf.Abs(transform.position.x - ball.transform.position.x) <= 1 || 
                Mathf.Abs(transform.position.y - ball.transform.position.y) <= 1)
                {
                   //TODO: Correct not adjacent bug and add direction human is facing logic 
                    
                    if(grabbedObject == null)
                    {
                        grabbedObject = ball;
                        grabbedObject.GetComponent<Rigidbody2D>().isKinematic = true;
                        grabbedObject.transform.position = transform.position;
                        grabbedObject.transform.SetParent(transform);
                        Debug.Log("Grabbed object = new ball");
                    }
                    else
                    {
                        //TODO: Check if ball can be dropped in position
                        //Add here check with backend to see whether robot is adjacent and can receive ball
                        grabbedObject.GetComponent<Rigidbody2D>().isKinematic = false;
                        grabbedObject.transform.position =  new Vector3(transform.position.x - 1, transform.position.y, 0);
                        grabbedObject.transform.SetParent(null);
                        grabbedObject = null;
                        Debug.Log("Grabbed object = ");
                    }
                    break;
                }
            }
        }
        
        /*if(Mathf.Abs(transform.position.x - grabbedObject.transform.position.x) == 1 || 
        Mathf.Abs(transform.position - grabbedObject.transform.position.x) == 1)
        //if(hitInfo.collider != null && hitInfo.collider.gameObject.layer == layerIndex)
        {
            Debug.Log("Entered if");

            if(Input.GetKeyDown(KeyCode.Space) && grabbedObject == null)
            {
                grabbedObject = hitInfo.collider.gameObject;
                grabbedObject.GetComponent<Rigidbody2D>().isKinematic = true;
                grabbedObject.transform.position = grabPoint.position;
                grabbedObject.transform.SetParent(transform);
                Debug.Log("Space Pressed, Grabbed object = null");
            }
            else if(Input.GetKeyDown(KeyCode.Space))
            {
                grabbedObject.GetComponent<Rigidbody2D>().isKinematic = false;
                grabbedObject.transform.SetParent(null);
                grabbedObject = null;
                Debug.Log("Space Pressed, Grabbed object = ");
            }
        }*/
    }
}
