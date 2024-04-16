using UnityEngine;
using System;
using System.IO;
using UnityEngine.UI;
using System.Collections;
using UnityEngine.Networking;
using UnityEngine.SceneManagement;
using System.Collections.Generic;
using Newtonsoft.Json;


public class LogManager : MonoBehaviour
{
    public string logID; 
    private string logFileName;
    public GameObject Button;
    public GameObject box_input;
    public GameObject txt;
    public string SOCKETS_IP;
    public int SERVER_PORT;
    public bool NGROK;

    public Dictionary<string, object> OldData = null;


    // Initialize the log file path
    private void Start()
    {   
        DontDestroyOnLoad(gameObject);
        SOCKETS_IP = "146.193.224.2";
        SERVER_PORT = 5000;
        NGROK = false;
    }

    // Method to write a log entry with a given log ID and additional data
    public void WriteLog(Dictionary<string, object> additionalData)
    {   
        additionalData["id"] = logID;
        additionalData["time"] = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"); 
        if (OldData != additionalData)
        {
            string jsonData = JsonConvert.SerializeObject(additionalData);
            StartCoroutine(SendPostRequest(jsonData));
        }
        else
        {
            Debug.Log("No Update.");
        }

        OldData = additionalData;
    }

    public void defineLogID(string id)
    {
        logID = id;
        StartCoroutine(SendPostRequest(id));
        Button.SetActive(true);
        box_input.SetActive(false);
        txt.SetActive(false);
    }

    public IEnumerator SendPostRequest(string jsonString)
    {
        string url = "http://" + SOCKETS_IP + ":" + SERVER_PORT.ToString() + "/";
        if(NGROK) {url = "https://78be-2001-690-2100-1041-9dd3-264d-56a0-1238.ngrok-free.app";}
        Debug.Log("URL: " + url);
        byte[] byteData = System.Text.Encoding.UTF8.GetBytes(jsonString);

        UnityWebRequest request = new UnityWebRequest(url, "POST");
        request.uploadHandler = new UploadHandlerRaw(byteData);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            Debug.Log("Response: " + request.downloadHandler.text);

        }
        else
        {
            Debug.LogError("Error sending HTTP POST request: " + request.error);
        }

    }
}