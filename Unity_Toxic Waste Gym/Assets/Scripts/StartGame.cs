using UnityEngine;
using System;
using System.Net;  
using System.Net.Sockets;  
using System.Text;  
using System.Threading;

public class StartGame : MonoBehaviour
{   
    public string SOCKETS_IP = "127.0.0.1";
    public int INBOUND_PORT = 20501;
    public string HOST = "0.tcp.eu.ngrok.io";
    public int PORT = 13943;
    public int OUTBOUND_PORT = 20500;
    public int SOCK_TIMEOUT = 5;
    public int BUFFER_SIZE = 1024;
    public Socket handler;
    public Socket outbound_socket;
    void Start()
    {   
        DontDestroyOnLoad(gameObject);
    }

    public void socketCode()
    {   
        byte[] bytes = new byte[BUFFER_SIZE];
        IPAddress ipAddress = IPAddress.Parse(SOCKETS_IP);
        IPEndPoint Outbound_localEndPoint = new IPEndPoint(ipAddress, OUTBOUND_PORT);
        //IPAddress ipAddress = Dns.GetHostAddresses(HOST)[0];
        //IPEndPoint Outbound_localEndPoint = new IPEndPoint(ipAddress, PORT);


        // Create Outbound Sockets
        outbound_socket = new Socket(ipAddress.AddressFamily, SocketType.Stream, ProtocolType.Tcp);

        outbound_socket.Connect(Outbound_localEndPoint);

        Debug.Log("Outbound socket connected to" + outbound_socket.RemoteEndPoint.ToString());
        
        string data = @"
            {
                ""command"": ""n_play"",
                ""data"":{""name"": ""human"", ""id"": 0, ""type"": 0}
            }<EOF>";


        string data2 = @"
            {
                ""command"": ""n_play"",
                ""data"":{""name"": ""robot"", ""id"": 1, ""type"": 1}
            }<EOF>";

        
        string data3 = @"
            {
                ""command"": ""start_game"",
                ""data"":""""
            }<EOF>";

        
        try {

            string jsonString = data.TrimEnd('<', 'E', 'O', 'F', '>');
            string jsonString2 = data2.TrimEnd('<', 'E', 'O', 'F', '>');
            string jsonString3 = data3.TrimEnd('<', 'E', 'O', 'F', '>');
            
            Debug.Log(jsonString);
            Debug.Log(jsonString2);
            Debug.Log(jsonString3);
            // Encode the data string into a byte array.
            byte[] msg = Encoding.ASCII.GetBytes(jsonString);
            byte[] msg2 = Encoding.ASCII.GetBytes(jsonString2);
            byte[] msg3 = Encoding.ASCII.GetBytes(jsonString3);

            // Send the data through the socket.
            int bytesSent = outbound_socket.Send(msg);
            Thread.Sleep(500);
            int bytesSent2 = outbound_socket.Send(msg2);
            Thread.Sleep(500);
            int bytesSent3 = outbound_socket.Send(msg3);

           /* // Release the socket.
            outbound_socket.Shutdown(SocketShutdown.Both);
            outbound_socket.Close();
            //outbound_socket.Disconnect(false);
            Debug.Log("Disconnected!");*/
        }
        catch (Exception e)
        {
            Debug.Log(e.ToString());
        }

    } 

}
