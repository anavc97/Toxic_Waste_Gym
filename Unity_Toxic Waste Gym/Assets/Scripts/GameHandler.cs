using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using CodeMonkey.Utils;
using UnityEditor.Scripting.Python;
using UnityEditor;
using System;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Diagnostics;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.IO;
using TMPro;

public class GameHandler : MonoBehaviour
{   
    public class GameData
    {   
        [JsonProperty("command")]
        public string Command { get; set; }
        
        [JsonProperty("data")]
        public Data Data { get; set; }
    }

    public class Data
    {
        public string Layout { get; set; }
        
        [JsonProperty("players")]
        public List<Player> Players { get; set; }

        [JsonProperty("objects")]
        public List<Ball> Objects { get; set; }

        [JsonProperty("finished")]
        public bool Finished { get; set; }

        [JsonProperty("timeout")]
        public bool Timeout { get; set; }

        [JsonProperty("game_id")]
        public int GameId { get; set; }

        [JsonProperty("score")]
        public float Score { get; set; }

        [JsonProperty("time_left")]
        public float TimeLeft { get; set; }

        [JsonProperty("game_time")]
        public float GameTime { get; set; }

        [JsonProperty("ticks")]
        public int Ticks { get; set; }
    }

    public class Player
    {   
        [JsonProperty("name")]
        public string Name { get; set; }

        [JsonProperty("position")]
        public List<int> Position { get; set; } = new List<int>();
        
        [JsonProperty("orientation")]
        public List<int> Orientation { get; set; } = new List<int>();
        
        [JsonProperty("held_object")]
        public List<Ball> HeldObject { get; set; }
        public float Health { get; set; }
    }

    public class Ball
    {   
        [JsonProperty("name")]
        public string Name { get; set; }

        [JsonProperty("position")]
        public List<int> Position { get; set; } = new List<int>();

        [JsonProperty("hold_state")]
        public int HoldState { get; set; }

        [JsonProperty("identified")]
        public bool Identified { get; set; }

        [JsonProperty("type")]
        public int Type { get; set; } //

        [JsonProperty("holding_player")]
        public object HoldingPlayer { get; set; }

        //Add ball type property
    }
    
    //[SerializeField] private HealthBar healthBar;
    [SerializeField] private Timer time;
    public float global_health;
    public string SOCKETS_IP = "127.0.0.1";
    public int INBOUND_PORT = 20501;
    public int OUTBOUND_PORT = 20500;
    public int SOCK_TIMEOUT = 5;
    public int BUFFER_SIZE = 1024;
    public Socket handler;
    public Socket outbound_socket;
    Thread SocketThreadIn;
    Thread SocketThreadOut;
    public GameData gameData;
    public InputHandler input_handler;
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
    public Stopwatch popUpStopWatch = new Stopwatch();
    
    void Awake()
    {   
        global_health = 1.0f;
        

        canvas = GameObject.FindWithTag("Canvas");
        /*FunctionPeriodic.Create(() => {
            if (global_health > 0){
                global_health -= 0.1f;
                healthBar.SetSize(global_health);            }

       
        }, 0.1f);*/
        
        ScoreScript.scoreValue = 0;
        popUp = GameObject.Find("PopUp").GetComponent<TMP_Text>();
        ballInteraction = GameObject.Find("red_1").GetComponent<BallInteraction>();
        gameRunning = true;
        StartServer();
        input_handler = GameObject.Find("human").GetComponent<InputHandler>();
    }

    void StartServer()
    {   
        SocketThreadIn = new Thread(socketInCode);
        SocketThreadIn.IsBackground = true;
        SocketThreadIn.Start();

        outbound_socket = GameObject.Find("SceneManager").GetComponent<StartGame>().outbound_socket;
        UnityEngine.Debug.Log("UNITY: Outbound Socket connected to" + outbound_socket.RemoteEndPoint.ToString());
        //socketOutCode();
    }

    void socketInCode()
    {
        IPAddress ipAddress = IPAddress.Parse(SOCKETS_IP);
        IPEndPoint Inbound_localEndPoint = new IPEndPoint(ipAddress, INBOUND_PORT);

        try {    
            // Create Outbound Sockets
            Socket inbound_socket = new Socket(ipAddress.AddressFamily, SocketType.Stream, ProtocolType.Tcp);
            
            // Set up Unity side connection: Listen to Backend Socket
            inbound_socket.Bind(Inbound_localEndPoint);
            inbound_socket.Listen(10);
            
            UnityEngine.Debug.Log("Waiting for a connection...");
            handler = inbound_socket.Accept();
            UnityEngine.Debug.Log("Server Started");
            UnityEngine.Debug.Log("Inbound socket started at " + SOCKETS_IP + " ," + INBOUND_PORT);

            while(gameRunning)
            {
                // Incoming data from the client.
                string data = null;
                byte[] bytes = null;

                while (true)
                {
                    bytes = new byte[BUFFER_SIZE];
                    int bytesRec = handler.Receive(bytes);
                    data += Encoding.ASCII.GetString(bytes, 0, bytesRec);
                    if (data.IndexOf("<EOF>") > -1)
                    {
                        break;
                    }

                    System.Threading.Thread.Sleep(1);
                }
                System.Threading.Thread.Sleep(1);

                //Trim data to JSON format
                string jsonData = data.TrimEnd('<', 'E', 'O', 'F', '>');
                // Read new state
                readState(jsonData);
            }
        }
        catch (Exception e)
        {
            UnityEngine.Debug.Log(e.ToString());
        }
    }

    void socketOutCode()
    {
        IPAddress ipAddress = IPAddress.Parse(SOCKETS_IP);
        IPEndPoint Outbound_localEndPoint = new IPEndPoint(ipAddress, OUTBOUND_PORT);

        try 
        {
            // Create Outbound Sockets
            outbound_socket = new Socket(ipAddress.AddressFamily, SocketType.Stream, ProtocolType.Tcp);
            
            // Connect to Remote EndPoint
            outbound_socket.Connect(Outbound_localEndPoint);

            UnityEngine.Debug.Log("UNITY: Outbound Socket connected to" + outbound_socket.RemoteEndPoint.ToString());
        }
        catch (Exception e)
        {
            UnityEngine.Debug.Log(e.ToString());
        }
    } 

    public void SendActionMessage(string jsonmsg)
    {   
        byte[] msg = Encoding.ASCII.GetBytes(jsonmsg);
        int bytesSent = outbound_socket.Send(msg);
        UnityEngine.Debug.Log("Sending action: " + jsonmsg);
    } 

    void Update()
    {   
        //if (global_health > 0) {update_Score_Health(50, global_health-0.00016f);}
        if (gameData != null)
        {   
            // Access the data as needed
            if (gameData.Command == "new_state")
            {   
                
                foreach (var player in gameData.Data.Players)
                {
                    if (player.Name == "human")
                    {   
                        GameObject player_obj = GameObject.Find(player.Name);
                        ActionRendering action = player_obj.GetComponent<ActionRendering>();

                        action.moveOrRotate(new Vector3(player.Position[1],14-player.Position[0],0), new Vector2(player.Orientation[1],-player.Orientation[0]));
                        if(player.HeldObject != null && player.HeldObject.Count() > 0)
                        {  
                            holdingBall = true;
                            previousHeldBallType = player.HeldObject[0].Type;
                        }
                        update_popUp();
                        
                        action.humanInteractWithBall(holdingBall);

                        previousHoldingBall = holdingBall;
                        holdingBall = false;

                        update_Score(gameData.Data.Score);
                    }
                    if (player.Name == "astro")
                    {
                        GameObject player_obj = GameObject.Find(player.Name);
                        ActionRenderingRobot action = player_obj.GetComponent<ActionRenderingRobot>();

                        action.moveOrRotateRobot(new Vector3(player.Position[1],14-player.Position[0],0), new Vector2(player.Orientation[1],-player.Orientation[0]));

                    }

                }
                foreach (var obj in gameData.Data.Objects)
                {
                    updateBallState(obj.Name, obj.HoldState, obj.Position, obj.Identified);    
                }  
            }
            else if (gameData.Command == "game_finished")
            {   
                UnityEngine.Debug.Log("Game Over!");
                UnityEngine.Debug.Log("Time remaining: " + time.timeRemaining);
                /*Transform panel = canvas.transform.Find("Panel");
                if (time.timeRemaining <1f)
                {
                    panel.GetComponent<TextMeshPro>().text = "Time is Up!\n Ready for the next level?";
                }*/
                
                canvas.GetComponent<Canvas>().enabled = true;
                gameOver = true;
            } 

            popUp_time += 1;

            if(popUp_time == 200) //200 frames later
            {
                popUp.text = "";
                popUp_time = 0;
            }
            

        }
    
    }

    void updateBallState(string objName, int status, List<int> position, bool id)
    {
        GameObject ball = GameObject.Find(objName);
        if (ball != null)
        {   
            ball.transform.position = new Vector3(position[1], 14-position[0], 0);

            if(status == 2)
            {
                ball.tag = "CollectedBall";
                Destroy(ball);
            }
            else if(status == 1)
            {
                ball.GetComponent<SpriteRenderer>().enabled = false;
            }
            else if(status == 0)
            {   
                ball.GetComponent<SpriteRenderer>().enabled = true;
            }

            //if (id)
            //{
            //    ballInteraction.StartCoroutine(ballInteraction.StartIdAnimation(ball));
               
            //}
            /*else
            {
                ball.tag = "Ball";
            }*/

        }
        
    }
    
    void update_Score(float score)
    {      
        //healthBar.SetSize(health);
        ScoreScript.scoreValue = score;

    }

    void update_popUp()
    {
        if(previousHoldingBall != holdingBall){ //Need to add checks everytime !holdingBall to see if it was received by astro
            if(previousHeldBallType == 1 && !holdingBall) //1=Green ball
            {
                popUp.text = "+2 points!";
                popUp.color = new Color32(92,255,51,255); //Light green 
                popUp_time = 0;
            }
            else if(previousHeldBallType == 2 && !holdingBall) //2=Yellow ball
            {
                popUp.text = "+10 points!";
                popUp.color = new Color32(40,191,0,255); //Green
                popUp_time = 0;
                timeHoldingYellowBall = 0;
                popUpStopWatch.Reset();
            }
            else if(previousHeldBallType == 3 && holdingBall) //3=Red ball
            {
                popUp.text = "-5 points!";
                popUp.color = new Color32(184,0,0,255);   //Red
                popUp_time = 0;
            }
        }
        
        if(previousHeldBallType == 2 && holdingBall)
        {
            if(!popUpStopWatch.IsRunning)
            {
                popUpStopWatch.Start();
            }
            if(popUpStopWatch.Elapsed.Seconds == 1 || timeHoldingYellowBall == 0)
            {
                timeHoldingYellowBall += 1;
                popUp.text = -(timeHoldingYellowBall * 2) + " seconds!";
                popUp.color = new Color32(168,147,0,255);  //Yellow
                popUpStopWatch.Reset();
            }
            popUp_time = 0;
        }
    }
    
    void readState(string data)
    {   
        // Deserialize the JSON content into a Game object
        gameData = JsonConvert.DeserializeObject<GameData>(data);
        input_handler.sendAction = true;
        
        //UnityEngine.Debug.Log("JSON RECEIVED: " + data);
        //UnityEngine.Debug.Log("Command: " + gameData.Command);

    }
    
    void stopInboundServer()
    {

        //stop thread
        if (SocketThreadIn != null)
        {
            SocketThreadIn.Abort();
        }

        handler.Shutdown(SocketShutdown.Both);
        handler.Close();
        handler.Disconnect(false);
        UnityEngine.Debug.Log("Inbound disconnected!");
        
    }
    
    void stopOutboundServer()
    {
        outbound_socket.Shutdown(SocketShutdown.Both);
        outbound_socket.Close();
        outbound_socket.Disconnect(false);
        UnityEngine.Debug.Log("Outbound disconnected!");            
    }
    
    void OnDisable()
    {   
        UnityEngine.Debug.Log("Disabled.");
        gameRunning = false;
        stopInboundServer();
        stopOutboundServer();
    }

}

