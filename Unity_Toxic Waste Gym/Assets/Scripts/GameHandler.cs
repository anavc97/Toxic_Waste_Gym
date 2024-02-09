using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using CodeMonkey.Utils;
using UnityEditor.Scripting.Python;
using UnityEditor;
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Newtonsoft.Json;
using System.IO;

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

        [JsonProperty("points")]
        public int Points { get; set; }

        [JsonProperty("game_time")]
        public double GameTime { get; set; }

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
        public object HeldObject { get; set; }
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

        [JsonProperty("holding_player")]
        public object HoldingPlayer { get; set; }
    }
    
    [SerializeField] private HealthBar healthBar;
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
    public GameObject canvas;
    
    void Awake()
    {   
        global_health = 1.0f;
        canvas = GameObject.FindWithTag("Canvas");
        Debug.Log("Canvas: " + canvas);
        /*FunctionPeriodic.Create(() => {
            if (global_health > 0){
                global_health -= 0.1f;
                healthBar.SetSize(global_health);            }

       
        }, 0.1f);*/
        
        ScoreScript.scoreValue = 0;
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
        Debug.Log("UNITY: Outbound Socket connected to" + outbound_socket.RemoteEndPoint.ToString());
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
            
            //Debug.Log("Waiting for a connection...");
            handler = inbound_socket.Accept();
            Debug.Log("Server Started");
            Debug.Log("Inbound socket started at " + SOCKETS_IP + " ," + INBOUND_PORT);

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
            Debug.Log(e.ToString());
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

            Debug.Log("UNITY: Outbound Socket connected to" + outbound_socket.RemoteEndPoint.ToString());
        }
        catch (Exception e)
        {
            Debug.Log(e.ToString());
        }
    } 

    public void SendActionMessage(string jsonmsg)
    {   
        byte[] msg = Encoding.ASCII.GetBytes(jsonmsg);
        int bytesSent = outbound_socket.Send(msg);
        Debug.Log("Sending action: " + bytesSent);
    } 

    void Update()
    {   
             
        if (global_health > 0) {update_Score_Health(50, global_health-0.00016f);}

        if (gameData != null)
        {   
            Debug.Log("Updating gameData: " + gameData.Command + " " + gameData.Data.Players[0].Name);
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
                        bool holding = player.HeldObject != null;
                        action.humanInteractWithBall(holding);

                        //update_Score_Health(gameData.Data.Points, 0.5f);
                    }

                }

                foreach (var obj in gameData.Data.Objects)
                {
                    updateBallState(obj.Name, obj.HoldState, obj.Position);
                    
                }    
            }
        }

        if (global_health <= 0)
        {
            canvas.GetComponent<Canvas>().enabled = true;
            gameOver = true;
        }
    
    }

    void updateBallState(string objName, int status, List<int> position)
    {
        GameObject ball = GameObject.Find(objName);
        if (ball != null)
        {   
            ball.transform.position = new Vector3(position[1], 14-position[0], 0);

            if(status == 2)
            {
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
        }
        
    }
    
    public void update_Score_Health(int score, float health)
    {      
        global_health = health;
        healthBar.SetSize(health);
        ScoreScript.scoreValue = score;

    }

    void createJSON(string myStringData)
    {

        string filePath = "state.json";

        using (StreamWriter file = File.CreateText(filePath))
        {
            JsonSerializer serializer = new JsonSerializer();
            serializer.Serialize(file, myStringData);
        }
       
    }
    
    void readState(string data)
    {   
        // Deserialize the JSON content into a Game object
        gameData = JsonConvert.DeserializeObject<GameData>(data);
        input_handler.sendAction = true;
        Debug.Log("JSON RECEIVED: " + data);

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
        Debug.Log("Inbound disconnected!");
        
    }
    
    void stopOutboundServer()
    {
        outbound_socket.Shutdown(SocketShutdown.Both);
        outbound_socket.Close();
        outbound_socket.Disconnect(false);
        Debug.Log("Outbound disconnected!");            
    }
    
    void OnDisable()
    {   
        Debug.Log("Disabled.");
        gameRunning = false;
        stopInboundServer();
        stopOutboundServer();
    }

}

