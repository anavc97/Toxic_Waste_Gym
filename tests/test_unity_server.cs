using System;  
using System.Net;  
using System.Net.Sockets;  
using System.Text;  
using System.Threading;
  
// Client app is the one sending messages to a Server/listener.   
// Both listener and client can send messages back and forth once a   
// communication is established.  
public class SocketClient  
{   
    public static string SOCKETS_IP = "127.0.0.1";
    public static int INBOUND_PORT = 20501;
    public static int OUTBOUND_PORT = 20501;
    public static int SOCK_TIMEOUT = 5;
    public static int BUFFER_SIZE = 1024;

    public static int Main(String[] args)  
    {  
        Console.WriteLine("Initializing");
        StartClient();  
        return 0;  
    }  
  
    public static void StartClient()  
    {  
        byte[] bytes = new byte[1024]; 
        IPAddress ipAddress = IPAddress.Parse(SOCKETS_IP); 
        string jsonString = @"
            {
                ""command"": ""debug"",
                ""data"":
                {
                    ""layout"": ""Intro"",
                    ""players"": 
                    {
                        ""human"": {
                            ""id"": 1,
                            ""position"": [13,1],
                            ""orientation"": [1,0],
                            ""ball in hand"": true,
                            ""health"": 0.5
                        },
                        ""robot"": {
                            ""id"": 2,
                            ""position"": [0,-3],
                            ""orientation"": [0,1],
                            ""ball in hand"": []
                        }
                    },
                    ""objects"": 
                    {
                        ""ball1"": {
                            ""position"": [1,1],
                            ""status"": ""free"",
                            ""held"": 1
                        },
                        ""ball2"": {
                            ""position"": [4,4],
                            ""status"": ""disposed"",
                            ""held"": ""None""
                        },
                        ""ball3"": {
                            ""position"": [1,5],
                            ""status"": ""held"",
                            ""held"": ""None""
                        },
                        ""ball4"": {
                            ""position"": [3,7],
                            ""status"": ""free"",
                            ""held"": ""None""
                        }
                    },
                    ""score"": 30
                }
            }<EOF>";
  
        try  
        {  
            // Connect to a Remote server  
            IPEndPoint Outbound_localEndPoint = new IPEndPoint(ipAddress, INBOUND_PORT);
  
            // Create a TCP/IP  socket.    
            Socket sender = new Socket(ipAddress.AddressFamily, SocketType.Stream, ProtocolType.Tcp);  
  
            // Connect the socket to the remote endpoint. Catch any errors.    
            try  
            {  
                // Connect to Remote EndPoint  
                sender.Connect(Outbound_localEndPoint);  
  
                Console.WriteLine("Socket connected to {0}",  
                    sender.RemoteEndPoint.ToString());  
  
                // Encode the data string into a byte array.    
                byte[] msg = Encoding.ASCII.GetBytes(jsonString);  
  
                // Send the data through the socket.    
                int bytesSent = sender.Send(msg);  
  
                // Receive the response from the remote device.    
                int bytesRec = sender.Receive(bytes);  
                Console.WriteLine("Echoed test = {0}",  
                    Encoding.ASCII.GetString(bytes, 0, bytesRec));  

                Thread.Sleep(10000);
                // Release the socket.    
                sender.Shutdown(SocketShutdown.Both);  
                sender.Close();  
  
            }  
            catch (ArgumentNullException ane)  
            {  
                Console.WriteLine("ArgumentNullException : {0}", ane.ToString());  
            }  
            catch (SocketException se)  
            {  
                Console.WriteLine("SocketException : {0}", se.ToString());  
            }  
            catch (Exception e)  
            {  
                Console.WriteLine("Unexpected exception : {0}", e.ToString());  
            }  
  
        }  
        catch (Exception e)  
        {  
            Console.WriteLine(e.ToString());  
        }  
    }  
} 