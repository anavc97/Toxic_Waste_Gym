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
    public List<float?> trustValueList = new List<float?> {null,null,null};
    public List<float?> manCheckList = new List<float?> {null,null,null};

    public bool ProlificIDCheck = false;

    // Initialize the log file path
    private void Start()
    {   
        DontDestroyOnLoad(gameObject);
        SOCKETS_IP = "146.193.224.2";
        SERVER_PORT = 2000;
        NGROK = 1;
    }

    void Update()
    {
        foreach(var trust in trustValueList)
        {
            UnityEngine.Debug.Log("Trust Values: " + trust);
        }
        foreach(var check in manCheckList)
        {
            UnityEngine.Debug.Log("Man Check Values: " + check);
        }
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
        if(id.Length != 24 && ProlificIDCheck)
        {
            StartCoroutine(PopErrorMessage());
        }
        else
        {   logID = id;
            errorMessage.gameObject.SetActive(false);
            StartCoroutine(SendPostRequest(id));
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
        string url = "http://" + SOCKETS_IP + ":" + SERVER_PORT.ToString() + "/";
        if(NGROK == 1) {url = "https://c4c96f0191a7.ngrok.app";SERVER_PORT = 2000;}
        if(NGROK == 2) {url = "https://113094bca628.ngrok.app";SERVER_PORT = 2100;}
        //if(NGROK == 3) {url = "https://7d515c71c6f4.ngrok.app";SERVER_PORT = 2200;}
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