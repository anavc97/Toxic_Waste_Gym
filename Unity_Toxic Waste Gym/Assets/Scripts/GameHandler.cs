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

public class GameData
{
    public string Command { get; set; }
    public Data Data { get; set; }
}

public class Data
{
    public string Layout { get; set; }
    public Dictionary<string, Player> Players { get; set; }
    public Dictionary<string, Ball> Objects { get; set; }
    public int Score { get; set; }
}

public class Player
{
    public int Id { get; set; }
    public List<int> Position { get; set; } = new List<int>();
    public List<int> Orientation { get; set; } = new List<int>();
    public List<string> BallInHand { get; set; } = new List<string>();
    public double Health { get; set; }
}

public class Ball
{
    public List<int> Position { get; set; } = new List<int>();
    public string Status { get; set; }
    public object Held { get; set; } = null;
}

public class GameHandler : MonoBehaviour
{
    [SerializeField] private HealthBar healthBar;
    public float global_health;
    public string SOCKETS_IP = "127.0.0.1";
    public int INBOUND_PORT = 20501;
    public int OUTBOUND_PORT = 20500;
    public int SOCK_TIMEOUT = 5;
    public int BUFFER_SIZE = 1024;
    public Socket handler;
    Thread SocketThread;
    public GameData gameData;
    
    void Start()
    {   
        FunctionPeriodic.Create(() => {
            if (global_health > 0){
                global_health -= 0.001f;
                healthBar.SetSize(global_health);            }

       
        }, 0.1f);
        
        ScoreScript.scoreValue = 0;
        StartServer();
    }

    void Update()
    {   
        // Access the data as needed
        Debug.Log($"Command: {gameData.Command}");
        Debug.Log($"Layout: {gameData.Data.Layout}");
        Debug.Log($"Score: {gameData.Data.Score}");

        foreach (var player in gameData.Data.Players)
        {
            Debug.Log($"\nPlayer: {player.Key}");
            Debug.Log($"ID: {player.Value.Id}");
            Debug.Log($"Position: [{player.Value.Position[0]}, {player.Value.Position[1]}]");
            Debug.Log($"Orientation: [{player.Value.Orientation[0]}, {player.Value.Orientation[1]}]");
            Debug.Log($"Ball in hand: {string.Join(", ", player.Value.BallInHand)}");
            Debug.Log($"Health: {player.Value.Health}");
        }

        foreach (var obj in gameData.Data.Objects)
        {
            Debug.Log($"\nObject: {obj.Key}");
            Debug.Log($"Position: [{obj.Value.Position[0]}, {obj.Value.Position[1]}]");
            Debug.Log($"Status: {obj.Value.Status}");
            Debug.Log($"Held: {obj.Value.Held}");
        }    
    }

    public void update_Score_Health(int score, float health)
    {   
        global_health = health;
        healthBar.SetSize(health);
        ScoreScript.scoreValue = score;
    }

    void StartServer()
    {   
        SocketThread = new Thread(socketCode);
        SocketThread.IsBackground = true;
        SocketThread.Start();

    }

    void socketCode()
    {
        IPAddress ipAddress = IPAddress.Parse(SOCKETS_IP);
        IPEndPoint Inbound_localEndPoint = new IPEndPoint(ipAddress, INBOUND_PORT);
        IPEndPoint Outbound_localEndPoint = new IPEndPoint(ipAddress, OUTBOUND_PORT);

        try {
            // Create Inbound/Outbound Sockets
            Socket inbound_socket = new Socket(ipAddress.AddressFamily, SocketType.Stream, ProtocolType.Tcp);
            Socket outbound_socket = new Socket(ipAddress.AddressFamily, SocketType.Stream, ProtocolType.Tcp);
            
            // Set up Unity side connection: Listen to Backend Socket
            inbound_socket.Bind(Inbound_localEndPoint);
            inbound_socket.Listen(10);
            Debug.Log("Waiting for a connection...");
            handler = inbound_socket.Accept();
            Debug.Log("Inbound socket started at " + SOCKETS_IP + " ," + INBOUND_PORT);

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

            byte[] msg = Encoding.ASCII.GetBytes(jsonData);
            handler.Send(msg);
        }
        catch (Exception e)
        {
            Debug.Log(e.ToString());
        }

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

    }
    
    void stopServer()
    {

        //stop thread
        if (SocketThread != null)
        {
            SocketThread.Abort();
        }

        if (handler != null && handler.Connected)
        {
            handler.Shutdown(SocketShutdown.Both);
            handler.Close();
            handler.Disconnect(false);
            Debug.Log("Disconnected!");
        }
    }
    
    void OnDisable()
    {   
        Debug.Log("Disabled.")
        stopServer();
    }

}

