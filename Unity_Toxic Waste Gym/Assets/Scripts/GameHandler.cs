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
    public List<Vector3> doorPositions = new List<Vector3>();
    public bool gameRunning;
    public bool gameOver = false;
    public bool holdingBall = false;
    public bool previousHoldingBall = false;
    public int previousHeldBallType = 0;
    public int popUp_time = 0;
    public int timeHoldingYellowBall = 0;
    public GameObject canvas;
    public TMP_Text popUp;
    public TMP_Text popTxt2;
    public GUIStyle guiStyle;
    public BallInteraction ballInteraction;
    public ScoreScript scoreScript;
    public Timer timerScript;
    public Stopwatch popUpStopWatch = new Stopwatch();
    public Stopwatch gameOverStopWatch = new Stopwatch();
    public GameObject humanPlayer;
    public GameObject astroPlayer;
    public GameObject[] balls;
    public GameObject heldBall;
    public string currentScene;
    public string layout;
    public Dictionary<string, float> yellowBallMap = new Dictionary<string, float>(); //(Ball name, time held)
    public GameObject logManager;
    public LogManager logger;
    private GameObject buttonObject;
    private int popUp_time_limit;
    private List<string> sceneList = new List<string>
            {"level_zero","level_one", "level_two", "level_three"};
    private Dictionary<string, object> additionalData = new Dictionary<string, object>();

    public Dictionary<string, int> TypeConverter = new Dictionary<string, int>{
        {"green", 1 },
        {"yellow", 2},
        {"red", 3}
        };
    
    private bool ball_idd;
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
        //canvas.GetComponent<Canvas>().enabled = false;
        logManager = GameObject.FindWithTag("Logger");
        logger = logManager.GetComponent<LogManager>();
        
        popUp = GameObject.Find("PopUp").GetComponent<TMP_Text>();
        ballInteraction = GameObject.Find("red_1").GetComponent<BallInteraction>();
        scoreScript = GameObject.Find("Score").GetComponent<ScoreScript>();
        timerScript = GameObject.Find("Timer").GetComponent<Timer>();
        humanPlayer = GameObject.Find("human");
        astroPlayer = GameObject.Find("astro");
        gameRunning = true;
        balls = GameObject.FindGameObjectsWithTag("Ball");
        scoreScript.scoreValue = 0;
        popUp_time_limit = 200;
        ball_idd = false;
        
        // Iterate through each ball in the list
        for (int i = 0; i < balls.Length; i++)
        {
            // Find the child object with the "Text" tag
            Transform textTransform = balls[i].transform.Find("Text");

            // Ensure the "Text" child exists
            if (textTransform != null)
            {
                // Get the TextMeshPro component
                TextMeshPro textMesh = textTransform.GetComponent<TextMeshPro>();

                // Update the text component with the index of the ball in the list
                textMesh.text = i.ToString();
            }
            else
            {
                UnityEngine.Debug.Log("Text child not found for ball: " + balls[i].name);
            }
        }
        
        // VARIABLES
        gameOver = false;
        holdingBall = false;
        previousHoldingBall = false;
        previousHeldBallType = 0;
        popUp_time = 0;
        timeHoldingYellowBall = 0;
        humanOrientation = new Vector2(0,-1); 
        doorPositions.Add(new Vector3(7, 15, 0));
        doorPositions.Add(new Vector3(6, 15, 0));
        doorPositions.Add(new Vector3(8, 15, 0));
        initializeYellowBallMap();
        StartCoroutine(Logging());
        guiStyle = new GUIStyle ();
        guiStyle.richText = true;
        if(SceneManager.GetActiveScene().name == "level_one"){scoreScript.globalScore = 0;}

    }
    private IEnumerator Logging()
    {
        while(!gameOver)
        {   
            additionalData = ConstructData();
            logger.WriteLog(additionalData);
            yield return new WaitForSeconds(0.2f);
        }
    }

    void Update()
    {   
        UnityEngine.Debug.Log("Game Over: " + gameOver);
        if(heldBall != null && heldBall.name.Split('_')[0] == "yellow" ) //Check if held ball is yellow
        {
            updatePopUp(heldBall,2);
        }

        popUp_time += 1;

        if((!timerScript.timeIsRunning && timerScript.timeRemaining != 150.0f || IsInVectorList(humanPlayer.transform.position, doorPositions)) && !gameOver)
        {   
            performHumanAction(0, 0, 1); // if game is done, drop ball - to account for time lost with yellow ball

            canvas.GetComponent<Canvas>().enabled = true;
            gameOver = true;
            astroPlayer.GetComponent<ActionRenderingRobot>().gameOverRobot = true;
            timerScript.gameOverTimer = true;
            
            GameObject panel = canvas.transform.Find("Panel").gameObject;
            GameObject gameOverText = panel.transform.Find("GameOver").gameObject;
            int bonusPoints = (int) Math.Max(0,timerScript.timeRemaining/10);
            if(SceneManager.GetActiveScene().name == "level_zero")
            {   
                gameOverText.GetComponent<TextMeshProUGUI>().text = "Tutorial complete! From now on, points will start counting to the end reward.\n Get ready for the first level...";
                gameOverStopWatch.Start();
            }
            else if(SceneManager.GetActiveScene().name == "level_one")
            {
                if(!timerScript.timeIsRunning){gameOverText.GetComponent<TextMeshProUGUI>().text = "Time ended!\n" + scoreScript.scoreValue + " out of 36 points acquired from balls.\nNo bonus points added for time left.\nLoading next level...";}
                else{gameOverText.GetComponent<TextMeshProUGUI>().text = "Level exited!\n" + scoreScript.scoreValue + " out of 36 points acquired from balls.\n" + bonusPoints + " bonus points added for time left." + "\nLoading next level...";}
                update_Score(bonusPoints);
                gameOverStopWatch.Start();
            }
            else if(SceneManager.GetActiveScene().name == "level_two")
            {   
                if(timerScript.timeRemaining <= 0){gameOverText.GetComponent<TextMeshProUGUI>().text = "Time ended!\n" + scoreScript.scoreValue + " out of 33 points acquired from balls.\nNo bonus points added for time left.\nLoading next level...";}
                else{gameOverText.GetComponent<TextMeshProUGUI>().text = "Level exited!\n" + scoreScript.scoreValue + " out of 33 points acquired from balls.\n" + bonusPoints + " bonus points added for time left." + "\nLoading next level...";}
                update_Score(bonusPoints);
                gameOverStopWatch.Start();
            }
            else if(SceneManager.GetActiveScene().name == "level_three")
            {   
                GameObject gameOverText2 = panel.transform.Find("GameOver2").gameObject;
                if(timerScript.timeRemaining <= 0){gameOverText.GetComponent<TextMeshProUGUI>().text = "Time ended!\n" + scoreScript.scoreValue + " out of 26 points acquired from balls.\nNo bonus points added for time left.";}
                else{gameOverText.GetComponent<TextMeshProUGUI>().text = "Level exited!\n" + scoreScript.scoreValue + " out of 26 points acquired from balls.\n" + bonusPoints + " bonus points added for time left.";
                     update_Score(bonusPoints);
                    if(logger.NGROK == 1){gameOverText2.GetComponent<TextMeshProUGUI>().text=  "Game Concluded! Final Score: " + scoreScript.globalScore + "\n End of game code: TX965U";}
                    else if(logger.NGROK == 2){gameOverText2.GetComponent<TextMeshProUGUI>().text=  "Game Concluded! Final Score: " + scoreScript.globalScore + "\n End of game code: VSWN20";}}
                
            }
        }

        if (gameOver){popUp_time_limit = 1000;}else{popUp_time_limit = 250;}
        if(popUp_time >= popUp_time_limit) //200 frames later
        {
            popUp.text = "";
            popTxt2.text = "";
            popUp_time = 0;
        }

        if(gameOverStopWatch.IsRunning && gameOverStopWatch.Elapsed.Seconds >= 8)
        {   
            int currentIndex = sceneList.IndexOf(SceneManager.GetActiveScene().name);  
            SceneManager.LoadScene(sceneList[currentIndex+1]);
            UnityEngine.Debug.Log("Loading scene: " + sceneList[currentIndex+1]);
           
        }
    }

    public Dictionary<string, object> ConstructData()
    {

        Dictionary<string, object> data = new Dictionary<string, object>();
        string b_held;
        List<string> b_disposed = new List<string>();

        if (heldBall == null){b_held = null;} else{b_held = heldBall.name;}
        Vector2 RobotOR = astroPlayer.GetComponent<ActionRenderingRobot>().astroOrientation;
                // Construct players list
        List<Dictionary<string, object>> players = new List<Dictionary<string, object>>();
        Dictionary<string, object> player1 = new Dictionary<string, object>();
        player1["name"] = "human";
        player1["position"] = new int[] { (int)humanPlayer.transform.position.x, (int)humanPlayer.transform.position.y };
        player1["orientation"] = new int[] { (int)humanOrientation.x, (int)humanOrientation.y };
        player1["held_object"] = b_held;
        
        
        Dictionary<string, object> player2 = new Dictionary<string, object>();
        player2["name"] = "astro";
        player2["position"] = new int[] { (int)astroPlayer.transform.position.x, (int)astroPlayer.transform.position.y };
        player2["orientation"] = new int[] { (int)RobotOR.x, (int)RobotOR.y };

        // Construct objects list
        List<Dictionary<string, object>> objects = new List<Dictionary<string, object>>();
        foreach (GameObject ball in balls)
        {   
            Dictionary<string, object> ballData = new Dictionary<string, object>();
            ballData["name"] = ball.name;
            ballData["hold_state"] = GetHoldStatus(ball);
            if(GetHoldStatus(ball) == 2) //ball was disposed, holding player is astro and position should be registered as -1 -1
            {
                ballData["position"] = new int[] {-1, -1};
                ballData["holding_player"] = "astro";
            }
            else
            {
                ballData["position"] = new int[] { (int)ball.transform.position.x, (int)ball.transform.position.y };
                ballData["holding_player"] = null;
            }
            if(GetHoldStatus(ball) == 1){ballData["position"] = new int[] { (int)humanPlayer.transform.position.x, (int)humanPlayer.transform.position.y };}

            ballData["identified"] = (ball.tag=="IDdBall");
            string type = ball.name.Split('_')[0];
            ballData["type"] = TypeConverter[type.ToLower()];
            if(ball.name == b_held){ballData["holding_player"] = "human";}
            objects.Add(ballData);
            if(ball.tag == "CollectedBall"){b_disposed.Add(ball.name);}
        }
        if (b_disposed.Count == 0){player2["held_object"] = null;}else{player2["held_object"] = b_disposed;}
        
        players.Add(player1);
        players.Add(player2);
        data["players"] = players;
        data["objects"] = objects;
        data["score"] = scoreScript.scoreValue;
        data["timeleft"] = timerScript.timeRemaining;
        data["layout"] = SceneManager.GetActiveScene().name;

        return data;

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
                    updateBallState(heldBall, 2, new Vector3(7,20,0));
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
                UnityEngine.Debug.Log("Ball disposed.");
                ball.GetComponent<SpriteRenderer>().enabled = true;
                ball.transform.Find("Text").GetComponent<TextMeshPro>().enabled = true;
                ball.tag = "CollectedBall";
            }
            else if(status == 1) //Ball grabbed
            {
                ball.GetComponent<SpriteRenderer>().enabled = false;
                ball.transform.Find("Text").GetComponent<TextMeshPro>().enabled = false;
            }
            else if(status == 0) //Ball dropped of
            {   
                ball.GetComponent<SpriteRenderer>().enabled = true;
                ball.transform.Find("Text").GetComponent<TextMeshPro>().enabled = true;
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
                    //popUp.text = ("+10 points!\n") + (-yellowBallMap[ball.name]) + (" seconds!");
                    //popUp.color = new Color32(168,147,0,255);  //Yellow;

                    popUp.text = "+10 points!";
                    popUp.color = new Color32(92,255,51,255); // green
                    popTxt2.text = (-yellowBallMap[ball.name]) + " seconds!";
                    popTxt2.color = new Color32(184,0,0,255); //Red

                    //popUp.color = new Color32(40,191,0,255); //Green
                    update_Score(10);
                    update_Timer(-yellowBallMap[ball.name]);
                    popUp_time = 0;
                    timeHoldingYellowBall = 0;
                }
                
                popUpStopWatch.Reset();
            }
            else if(ballType == "yellow" && status == 2) //Grabbed yellow ball
            {
                if(!popUpStopWatch.IsRunning)
                {
                    popUpStopWatch.Start();
                }
                if(popUpStopWatch.Elapsed.TotalSeconds >= 1 || timeHoldingYellowBall == 0)
                {
                    timeHoldingYellowBall += 1;
                    //popUp.text = -(timeHoldingYellowBall * 2) + " seconds!";
                    //popUp.text = -(timeHoldingYellowBall) + " seconds!";
                    //popUp.color = new Color32(168,147,0,255);  //Yellow
                    update_effectiveTimer(-3);
                    yellowBallMap[ball.name] += 3;
                    popUpStopWatch.Reset();
                }
                popUp_time = 0;
            }
        }
    }

    void update_Score(float score)
    {      
        scoreScript.scoreValue += score;
        scoreScript.globalScore += score;
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

    int GetHoldStatus(GameObject ball)
    {
        if (ball.transform.position.x == 7f && ball.transform.position.y == 20f) {return 2;} // Disposed
        else if(!ball.GetComponent<SpriteRenderer>().enabled){return 1;} // Holding
        else{return 0;} // free
    }

    bool IsInVectorList(Vector3 position, List<Vector3> vectorList)
    {
        foreach (Vector3 vector in vectorList)
        {
            if (vector == position)
            {   
                return true;
            }
        }
        return false;
    }
}

