using UnityEngine;
using System;
using System.IO;
using UnityEngine.UI;
using System.Collections;
using UnityEngine.Networking;
using UnityEngine.SceneManagement;
using System.Collections.Generic;
using Newtonsoft.Json;
using TMPro;
using Unity.VisualScripting;
using System.Text.RegularExpressions;



public class LogManager : MonoBehaviour
{
    public string logID; 
    private string logFileName;
    public GameObject Button;
    public GameObject box_input;
    public GameObject txt;
    public GameObject errorMessage;
    public string SOCKETS_IP;
    public int SERVER_PORT;
    public int NGROK;
    public GameObject TotTimer;

    public Dictionary<string, object> OldData = null;
    public List<float?> trustValueList = new List<float?> {null,null,null,null};
    public List<float?> manCheckList = new List<float?> {null,null,null,null};

    public bool ProlificIDCheck = false;

    private ActionRenderingRobot robotController;

    // Initialize the log file path
    private void Start()
    {   
        DontDestroyOnLoad(gameObject);
        SOCKETS_IP = "127.0.0.1";
        SERVER_PORT = 2000;
        NGROK = 3;
        if(SceneManager.GetActiveScene().name != "Intro" && SceneManager.GetActiveScene().name != "Tutorial 3"){robotController = GameObject.Find("astro").GetComponent<ActionRenderingRobot>();}
    }

    void Update()
    {
        /*foreach(var trust in trustValueList)
        {
            UnityEngine.Debug.Log("Trust Values: " + trust);
        }
        foreach(var check in manCheckList)
        {
            UnityEngine.Debug.Log("Man Check Values: " + check);
        }*/
    }
    // Method to write a log entry with a given log ID and additional data
    public void WriteLog(Dictionary<string, object> additionalData)
    {   
        additionalData["trustValues"] = trustValueList;
        additionalData["ManCheckValues"] = manCheckList;
        additionalData["id"] = logID;
        additionalData["time"] = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");
        TotTimer = GameObject.Find("Tutorial Timer"); 
        if(TotTimer!=null)
        {
            additionalData["tutorial_time"] = TotTimer.GetComponent<TutorialTimer>().tutorial_timer;
            Debug.Log("destroied");
            Destroy(TotTimer);
        }

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
        string idMessage;
        if(id.Length != 24 && ProlificIDCheck)
        {
            StartCoroutine(PopErrorMessage());
        }
        else
        {   logID = id + "_" + DateTime.Now.ToString("MMddHHmmss");
            errorMessage.gameObject.SetActive(false);
            idMessage = "{\"logid\": \"" + id + "\"}";
            StartCoroutine(SendPostRequest(idMessage));
            Button.SetActive(true);
            box_input.SetActive(false);
            txt.SetActive(false);
        }
    }
    public IEnumerator PopErrorMessage()
    {
        errorMessage.SetActive(true);

        yield return new WaitForSeconds(4f);

        errorMessage.SetActive(false);
    }

    public IEnumerator SendPostRequest(string jsonString)
    {
        Debug.Log("Sending Log");
        string url = "http://" + SOCKETS_IP + ":" + SERVER_PORT.ToString() + "/";
        if(NGROK == 1) {url = "https://5e4a3a106185.ngrok.app";SERVER_PORT = 2000;}
        if(NGROK == 2) {url = "https://095cfe8e834d.ngrok.app";SERVER_PORT = 2100;}
        if(NGROK == 3) {url = "https://8a5393958d89.ngrok.app";SERVER_PORT = 2200;}
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

        public IEnumerator SendActionRequest(string jsonString)
    {   
        Debug.Log("Message: " + jsonString);
        robotController = GameObject.Find("astro").GetComponent<ActionRenderingRobot>();

        string url = "http://" + SOCKETS_IP + ":" + SERVER_PORT.ToString() + "/";
        if(NGROK == 1) {url = "https://5e4a3a106185.ngrok.app";SERVER_PORT = 2000;}
        if(NGROK == 2) {url = "https://095cfe8e834d.ngrok.app";SERVER_PORT = 2100;}
        if(NGROK == 3) {url = "https://8a5393958d89.ngrok.app";SERVER_PORT = 2200;}
        
        byte[] byteData = System.Text.Encoding.UTF8.GetBytes(jsonString);

        UnityWebRequest request = new UnityWebRequest(url, "POST");
        request.uploadHandler = new UploadHandlerRaw(byteData);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();
        Debug.Log("Server Reply Action: " + request.downloadHandler.text);
        
        if (request.result == UnityWebRequest.Result.Success)
        {
            Match match = Regex.Match(request.downloadHandler.text, @"\d+$");
            
            if (match.Success)
            {
                Debug.Log("Robot action changed: " + int.Parse(match.Value));
                robotController.action = int.Parse(match.Value);

            }
        }
        else
        {
            Debug.LogError("Error sending HTTP POST request: " + request.error);
        }

    }
}