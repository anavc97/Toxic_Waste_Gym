using System.Collections;
using UnityEngine;
using Newtonsoft.Json;


public class InputHandler : MonoBehaviour
{
    
    private bool actionExecuted = false;
    private bool waiting = false;
    private bool gameOver;
    private float mov_x = 0;
    private float mov_y = 0;
    private int handleBall = 0;
    private GameHandler gameHandler;


    void Start()
    {
        gameHandler =  GameObject.Find("GameHandler").GetComponent<GameHandler>();
    }

    // Update is called once per frame
    void Update()
    {   
        gameOver = gameHandler.gameOver;
        
        if (!waiting && !gameOver)
        {
            if(Input.GetKeyDown(KeyCode.RightArrow) || Input.GetKeyDown(KeyCode.LeftArrow))
            {   
                mov_x = Input.GetAxisRaw("Horizontal");
                actionExecuted = true;
                if (!waiting)
                {
                    StartCoroutine(Wait());
                }
            }
            else if(Input.GetKeyDown(KeyCode.UpArrow) || Input.GetKeyDown(KeyCode.DownArrow))
            {   
                mov_y = Input.GetAxisRaw("Vertical");
                actionExecuted = true;
                if (!waiting)
                {
                    StartCoroutine(Wait());
                }
            }
            else if(Input.GetKeyDown(KeyCode.Space))
            {   
                handleBall = 1;
                actionExecuted = true;
                if (!waiting)
                {
                    StartCoroutine(Wait());
                }
            }

            if(actionExecuted)
            {   
                actionExecuted = false;
                gameHandler.performHumanAction(mov_x, mov_y, handleBall);
                mov_x = 0;
                mov_y = 0;
                handleBall = 0;
            }
        }
    }

    private IEnumerator Wait()
    {
      waiting = true;
      yield return new WaitForSeconds(0.2f);
      waiting = false;
    }

}
