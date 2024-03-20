using UnityEngine;
using TMPro;
using System.Diagnostics;

public class Timer : MonoBehaviour
{   
    public float timeRemaining;
    public bool timeIsRunning = true;
    public bool gameOverTimer = false;
    public TMP_Text timeText;
    public Stopwatch stopWatch;

    void Start()
    {
        timeIsRunning = true;
        timeRemaining = 150.0f;
        stopWatch = new Stopwatch();
    }

    void Update()
    {   
        if(timeIsRunning && !gameOverTimer)
        {
            if(!stopWatch.IsRunning){stopWatch.Start();}
            if(stopWatch.Elapsed.Seconds >= 1)
            {
                timeRemaining -= 1;
                DisplayTime(timeRemaining);
                stopWatch.Reset();
            }
            if(timeRemaining < 0){timeIsRunning = false;}
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


