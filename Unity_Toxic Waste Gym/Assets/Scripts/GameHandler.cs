using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Diagnostics;
using Newtonsoft.Json;
using TMPro;
using UnityEngine.SceneManagement;

public class GameHandler : MonoBehaviour
{   
    
    //[SerializeField] private Timer time;
    
    public Vector3 humanPosition;
    public Vector2 humanOrientation;
    public Vector3 astroPosition;
    public Vector3 doorPosition;
    public bool gameRunning;
    public bool gameOver = false;
    public bool holdingBall = false;
    public bool previousHoldingBall = false;
    public int previousHeldBallType = 0;
    public int popUp_time = 0;
    public int timeHoldingYellowBall = 0;
    public GameObject canvas;
    public TMP_Text popUp;
    public BallInteraction ballInteraction;
    public ScoreScript scoreScript;
    public Timer timerScript;
    public Stopwatch popUpStopWatch = new Stopwatch();
    public Stopwatch gameOverStopWatch = new Stopwatch();
    public GameObject humanPlayer;
    public GameObject astroPlayer;
    public GameObject heldBall;
    public string currentScene;
    public string layout;
    public Dictionary<string, float> yellowBallMap = new Dictionary<string, float>(); //(Ball name, time held)
    public GameObject logManager;
    public LogManager logger;
    
    void Awake()
    {        
        //DontDestroyOnLoad(gameObject);
    }

    void Start()
    {   
        //DontDestroyOnLoad(gameObject);
        currentScene = SceneManager.GetActiveScene().name;
        layout = currentScene;
        canvas = GameObject.FindWithTag("Canvas");
        canvas.GetComponent<Canvas>().enabled = false;
        logManager = GameObject.FindWithTag("Logger");
        logger = logManager.GetComponent<LogManager>();
        
        
        //ScoreScript.scoreValue = 0;
        popUp = GameObject.Find("PopUp").GetComponent<TMP_Text>();
        ballInteraction = GameObject.Find("red_1").GetComponent<BallInteraction>();
        scoreScript = GameObject.Find("Score").GetComponent<ScoreScript>();
        timerScript = GameObject.Find("Timer").GetComponent<Timer>();
        humanPlayer = GameObject.Find("human");
        astroPlayer = GameObject.Find("astro");
        gameRunning = true;
        
        
        // VARIABLES
        gameOver = false;
        holdingBall = false;
        previousHoldingBall = false;
        previousHeldBallType = 0;
        popUp_time = 0;
        timeHoldingYellowBall = 0;
        humanOrientation = new Vector2(0,-1); 
        doorPosition = new Vector3(7,14,0);
        initializeYellowBallMap();
        StartCoroutine(Logging());
    }
    private IEnumerator Logging()
    {
        while(!gameOver)
        {   
            string b_held;
            if (heldBall == null){b_held = null;} else{b_held = heldBall.name;}
            string log = $@"""name"": ""human"", ""position"":{humanPlayer.transform.position}, ""orientation"": {humanOrientation}, ""held_object"": {b_held}";
            log = log + "\n" + $@"""name"": ""robot"", ""position"":{astroPlayer.transform.position}, ""orientation"": {astroPlayer.GetComponent<ActionRenderingRobot>().astroOrientation}";

            logger.WriteLog(log);
            yield return new WaitForSeconds(0.2f);
        }
    }

    void Update()
    {   
        if(heldBall != null && heldBall.name.Split('_')[0] == "yellow" ) //Check if held ball is yellow
        {
            updatePopUp(heldBall,2);
        }

        popUp_time += 1;
        if(popUp_time >= 200) //200 frames later
        {
            popUp.text = "";
            popUp_time = 0;
        }

        if((!timerScript.timeIsRunning || humanPlayer.transform.position == doorPosition) && !gameOver)
        {
            canvas.GetComponent<Canvas>().enabled = true;
            gameOver = true;
            astroPlayer.GetComponent<ActionRenderingRobot>().gameOverRobot = true;
            timerScript.gameOverTimer = true;
            
            GameObject panel = canvas.transform.Find("Panel").gameObject;
            GameObject gameOverText = panel.transform.Find("GameOver").gameObject;
            if(SceneManager.GetActiveScene().name == "level_one")
            {
                if(!timerScript.timeIsRunning){gameOverText.GetComponent<TextMeshProUGUI>().text = "Time ended!\n" + scoreScript.scoreValue + " out of 36 points acquired.\n Loading next level...";}
                else{gameOverText.GetComponent<TextMeshProUGUI>().text = "Level exited!\n" + scoreScript.scoreValue + " out of 36 points acquired.\n Loading next level...";}
                gameOverStopWatch.Start();
            }
            else
            {
                if(timerScript.timeRemaining <= 0){gameOverText.GetComponent<TextMeshProUGUI>().text = "Time ended!\n" + scoreScript.scoreValue + " out of 33 points acquired.\n Game concluded.";}
                else{gameOverText.GetComponent<TextMeshProUGUI>().text = "Level exited!\n" + scoreScript.scoreValue + " out of 33 points acquired.\n  Game concluded.";}
            }
        }

        if(gameOverStopWatch.IsRunning && gameOverStopWatch.Elapsed.Seconds >= 12)
        {
            SceneManager.LoadScene("level_two");
            logger.WriteLog("############ LEVEL 2 ############");
        }
    }

    
    public void initializeYellowBallMap()
    {
        GameObject[] balls = GameObject.FindGameObjectsWithTag("Ball");
        foreach (GameObject ball in balls){   
            if(ball != null && ball.name.Split('_')[0] == "yellow")
            {
                yellowBallMap[ball.name] = 0;
            }
        }
    }   


    public void performHumanAction(float mov_x, float mov_y, int handleBall)
    {
        getPlayerPositions();
        if(handleBall != 0)
        {
            bool holdingBall = humanPlayer.GetComponent<ActionRendering>().getHasBall();
            if(holdingBall) //Might stop holding a ball
            {
                Vector3 newBallPosition = new Vector3(humanPosition.x + humanOrientation.x, humanPosition.y + humanOrientation.y, 0);
                //if(Vector3.Distance(humanPosition, astroPosition) <= Mathf.Sqrt(2)){ 
                if(newBallPosition == astroPosition) //Ball handed to astro (human must be facing it)
                {
                    updateBallState(heldBall, 2, newBallPosition);
                    updatePopUp (heldBall, 0); //0=Point update
                    humanPlayer.GetComponent<ActionRendering>().humanInteractWithBall();
                    heldBall = null;
                }
                else if(ballInteraction.checkPositionVacancy(newBallPosition)) //Make sure ball not dropped on top of other ball
                {  
                    updateBallState(heldBall, 0, newBallPosition);
                    updatePopUp(heldBall, 1); //1=might have stopped holding yellow ball
                    humanPlayer.GetComponent<ActionRendering>().humanInteractWithBall();
                    heldBall = null;
                }
            }
            else //Migjt start holding a ball
            {
                GameObject closestBall = ballInteraction.findClosestBall(ballInteraction.allBalls, humanPosition, humanOrientation, true);
                if(closestBall != null)
                {
                    heldBall = closestBall;
                    updateBallState(heldBall, 1, new Vector3(0,0,0));
                    humanPlayer.GetComponent<ActionRendering>().humanInteractWithBall();
                    updatePopUp(heldBall, 2); //2=might have started holding yellow ball
                }
                
            }

        }
        else
        {
            Vector3 move = humanPosition;
            Vector2 orientation = new Vector2(0,0);
            if(mov_x != 0)
            {
                move.x = move.x + mov_x;
                orientation.x = mov_x;
            }
            else if(mov_y != 0)
            {
                move.y = move.y + mov_y;
                orientation.y = mov_y;
            }
            humanPlayer.GetComponent<ActionRendering>().moveOrRotate(move,orientation);
            humanOrientation = orientation;
        }
    }

    void updateBallState(GameObject ball, int status, Vector3 newPosition)
    {
        if (ball != null)
        {   
            ball.transform.position = newPosition;
            if(status == 2) //Ball disposed of
            {
                ball.tag = "CollectedBall";
                Destroy(ball);
            }
            else if(status == 1) //Ball grabbed
            {
                ball.GetComponent<SpriteRenderer>().enabled = false;
            }
            else if(status == 0) //Ball dropped of
            {   
                ball.GetComponent<SpriteRenderer>().enabled = true;
            }
        }
        else{UnityEngine.Debug.Log("Ball to update state is null (not recognized as held)");}
        
    }
    
    void updatePopUp(GameObject ball, int status)
    {
        if(ball != null)
        {   
            string ballType = ball.name.Split('_')[0];
            if(ballType == "green" && status == 0)
            {
                popUp.text = "+3 points!";
                popUp.color = new Color32(92,255,51,255); //Light green 
                update_Score(3);
                popUp_time = 0;
            }
            else if(ballType == "red" && status == 0)
            {
                popUp.text = "-8 points!";
                popUp.color = new Color32(184,0,0,255); //Red
                update_Score(-8);
                popUp_time = 0;
            }
            else if(ballType == "yellow" && status <= 1) //Dropped yellow ball
            {
                if(status == 0) //Handed ball to robot
                {
                    popUp.text = ("+10 points!\n") + (-yellowBallMap[ball.name]) + (" seconds!");
                    popUp.color = new Color32(168,147,0,255);  //Yellow;

                    //popUp.color = new Color32(40,191,0,255); //Green
                    update_Score(10);
                    update_Timer(-yellowBallMap[ball.name]);
                    popUp_time = 0;
                }
                else{popUp_time = 150;}
                timeHoldingYellowBall = 0;
                popUpStopWatch.Reset();
            }
            else if(ballType == "yellow" && status == 2) //Grabbed yellow ball
            {
                if(!popUpStopWatch.IsRunning)
                {
                    popUpStopWatch.Start();
                }
                if(popUpStopWatch.Elapsed.Seconds >= 1 || timeHoldingYellowBall == 0)
                {
                    timeHoldingYellowBall += 1;
                    //popUp.text = -(timeHoldingYellowBall * 2) + " seconds!";
                    //popUp.text = -(timeHoldingYellowBall) + " seconds!";
                    //popUp.color = new Color32(168,147,0,255);  //Yellow
                    update_effectiveTimer(-3);
                    yellowBallMap[ball.name] += 3;
                    //UnityEngine.Debug.Log("Yellow ball named: " + ball.name + "held for " + yellowBallMap[ball.name] + "secs");
                    popUpStopWatch.Reset();
                }
                popUp_time = 0;
            }
        }
    }

    void update_Score(float score)
    {      
        scoreScript.scoreValue += score;
    }

    void update_Timer(float time)
    {      
        //timerScript.timeChangedCounter = 0;
        timerScript.timeRemaining += time;
        timerScript.DisplayTime(timerScript.timeRemaining);
    }

    void update_effectiveTimer(float time)
    {
        timerScript.effectiveTimeRemaining += time;
    }
    
    void getPlayerPositions()
    {
        humanPosition = humanPlayer.transform.position;
        astroPosition = astroPlayer.transform.position;
    }

}

