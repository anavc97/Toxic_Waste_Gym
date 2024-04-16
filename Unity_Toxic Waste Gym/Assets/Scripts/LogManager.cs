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

    public Dictionary<string, object> OldData = null;


    // Initialize the log file path
    private void Start()
    {   
        DontDestroyOnLoad(gameObject);
        SOCKETS_IP = "127.0.0.1";
        SERVER_PORT = 5000;
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