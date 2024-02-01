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

public class StartGame : MonoBehaviour
{   
    public string SOCKETS_IP = "127.0.0.1";
    public int INBOUND_PORT = 20501;
    public int OUTBOUND_PORT = 20500;
    public int SOCK_TIMEOUT = 5;
    public int BUFFER_SIZE = 1024;
    public Socket handler;
    Thread SocketThread;
    void Start()
    {

    }

    void Update()
    {
        
    }

    public void StartServer()
    {   
        SocketThread = new Thread(socketCode);
        SocketThread.IsBackground = true;
        SocketThread.Start();

    }


    public void socketCode()
    {
        byte[] bytes = new byte[BUFFER_SIZE];
        IPAddress ipAddress = IPAddress.Parse(SOCKETS_IP);
        IPEndPoint Outbound_localEndPoint = new IPEndPoint(ipAddress, OUTBOUND_PORT);
        string data = @"
            {
                ""command"": ""start_game"",
                ""data"":{ }
            }<EOF>";

        try {
            // Create Inbound/Outbound Sockets
            Socket outbound_socket = new Socket(ipAddress.AddressFamily, SocketType.Stream, ProtocolType.Tcp);

            // Connect to Remote EndPoint
            outbound_socket.Connect(Outbound_localEndPoint);

            Debug.Log("Socket connected to" + outbound_socket.RemoteEndPoint.ToString());

            string jsonString = data.TrimEnd('<', 'E', 'O', 'F', '>');
            // Encode the data string into a byte array.
            byte[] msg = Encoding.ASCII.GetBytes(jsonString);

            // Send the data through the socket.
            int bytesSent = outbound_socket.Send(msg);

            // Receive the response from the remote device.
            int bytesRec = outbound_socket.Receive(bytes);
            Debug.Log("Echoed test = " + Encoding.ASCII.GetString(bytes, 0, bytesRec));

            // Release the socket.
            outbound_socket.Shutdown(SocketShutdown.Both);
            outbound_socket.Close();
            outbound_socket.Disconnect(false);
            Debug.Log("Disconnected!");
        }
        catch (Exception e)
        {
            Debug.Log(e.ToString());
        }

    } 


}
