using UnityEngine;
using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
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
        string logEntry = $@"""Time"": ""[{DateTime.Now.ToString("hh:mm:ss tt")}]"", ""State"": ""[{additionalData}]"", ";
        //Debug.Log(logFilePath);
        // Append the log entry to the log file
        File.AppendAllText(logFilePath, logEntry + Environment.NewLine);
    }

    public void printLevelStats(List<float> playerDistances, int totalTimeYellowBalls, int totalTimeAllBalls, string levelEndedReason, int nBallsIdd, Dictionary<string, Vector3> ballPositionsMap)
    {
        string statsSeperator = "]" + "----------------------------------------------- Stats -----------------------------------------------" + "\n";
        File.AppendAllText(logFilePath, statsSeperator);

        double playerDistanceAvg = playerDistances.Average();
        double playerDistanceStandardDeviation = Math.Sqrt(playerDistances.Average(v => Math.Pow(v-playerDistanceAvg, 2)));
        double playerDistanceStandardError = playerDistanceStandardDeviation / Math.Sqrt(playerDistances.Count());
        
        string log = $@"""playerDistAvg"": ""{playerDistanceAvg}"", ""playerDistStandardDeviation"": ""{playerDistanceStandardDeviation}"", ""playerDistStandardError"": ""{playerDistanceStandardError}"", ";
        log = log + "\n" + $@"""levelOverReason"": ""{levelEndedReason}"", ""timeWithBalls"": ""{totalTimeAllBalls}"", ""timeWithYellowBalls"": ""{totalTimeYellowBalls}"", ""NOfBallsIdd"": ""{nBallsIdd}"", ";

        List<string> disposedBalls = new List<string>();
        foreach(var item in ballPositionsMap)
        {
            if(item.Value == new Vector3(-1,-1,0)){disposedBalls.Add(item.Key);}
        }
        log = log + "\n" + $@"""disposedBalls"": ""[{string.Join(", ", disposedBalls)}]""";
        
        File.AppendAllText(logFilePath, log + Environment.NewLine);
    }

    public void changeToLevel2()
    {
        string lvlSeparator = "###################################################################################################################" + "\n" + "################################################ LEVEL 2 #######################################################" + "\n" + "##################################################################################################################" + "\n" + "[";
        File.AppendAllText(logFilePath, lvlSeparator);
    }

    public void changeToLevel3()
    {
        string lvlSeparator = "###################################################################################################################" + "\n" + "################################################ LEVEL 3 #######################################################" + "\n" + "##################################################################################################################" + "\n" + "[";
        File.AppendAllText(logFilePath, lvlSeparator);
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
        string beginningOfFile = $@"""LogID"": ""{logID}""" + "\n" + "[";
        File.AppendAllText(logFilePath, beginningOfFile);
    }
}