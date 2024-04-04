using UnityEngine;
using System;
using System.IO;
using UnityEngine.UI;

public class LogManager : MonoBehaviour
{
    private string logFilePath; // Full path to the log file
    public string logID; 
    private string logFileName;
    public GameObject Button;
    public GameObject box_input;
    public GameObject txt;
    // Initialize the log file path
    private void Start()
    {   
        DontDestroyOnLoad(gameObject);
    }

    // Method to write a log entry with a given log ID and additional data
    public void WriteLog(string additionalData)
    {
        // Create a new log entry with current date and time, log ID, and additional data
        string logEntry = $"[{DateTime.Now}] {logID}: {additionalData}";
        //Debug.Log(logFilePath);
        // Append the log entry to the log file
        File.AppendAllText(logFilePath, logEntry + Environment.NewLine);
    }

    public void defineLogID(string id)
    {
        logID = id;
        logFileName = $"DataLogs/log_{logID}.txt";
        //Debug.Log("data path: " + Application.dataPath);
        //logFilePath = Application.dataPath + logFileName;
        logFilePath = logFileName;
        Button.SetActive(true);
        box_input.SetActive(false);
        txt.SetActive(false);
    }
}