using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BallInteraction : MonoBehaviour
{
    public Transform movePoint;
    //public Transform rayPoint;
    //public float rayDistance;

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
       
    }
        
    //Incomplete and not necessary
    public void changeBallStatus(string ballId, int newStatus)
    {
        foreach (GameObject ball in GameObject.FindGameObjectsWithTag("Ball"))
            {
                if(ball.name == ballId)
                {   
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
}
