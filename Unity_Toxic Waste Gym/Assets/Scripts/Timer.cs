using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class Timer : MonoBehaviour
{   
    public float timeRemaining;
    public bool timeIsRunning = true;
    public TMP_Text timeText;
    public GameHandler.GameData gameData;
    public GameObject gHandler;
    public float gameTime; 

    void Start()
    {
        timeIsRunning = true;
        timeRemaining = 120.0f;
        gHandler = GameObject.Find("GameHandler");
        gameData = gHandler.GetComponent<GameHandler>().gameData;
        Debug.Log(" -- " +gHandler + gameData);
    }

    void Update()
    {   
        gameData = gHandler.GetComponent<GameHandler>().gameData;
        Debug.Log(" -- " +gHandler + gameData);
        gameTime = gameData.Data.TimeLeft;

        if(timeIsRunning)
        {   
            
            if (timeRemaining >= 0)
            {
                timeRemaining = gameTime;
                DisplayTime(timeRemaining);
            }
            else
            {   
                timeIsRunning = false;
            }
        }
    }

    void DisplayTime (float timeToDisplay)
    {
        timeToDisplay +=1;
        float minutes = Mathf.FloorToInt(timeToDisplay/60);
        float seconds = Mathf.FloorToInt(timeToDisplay % 60);
        timeText.text = string.Format("Time Left: \n{0:00}:{1:00}", minutes, seconds);
    }
}


