using System.Collections;
using System.Collections.Generic;

using UnityEngine;
using TMPro;
using System.Threading;

public class BallInteraction : MonoBehaviour
{
    public GameObject historyChat;
    public GameObject currentChat;
    private GameObject bubble;
    private GameObject load;
    private GameObject text;

    public List<string> BallsIdentified = new List<string>();

    private int currentChatNumber = 1;

    void Start()
    {
        historyChat = GameObject.Find("HistoryChat");
    }

    void Update()
    {
       
    }

    public void StartIdAnimation(GameObject ball)
    {
        if(BallsIdentified.Contains(ball.name)) //Check if ball had already been identified before
        {
            return;
        }
        string type = ball.name.Split('_')[0];
        Debug.Log("Ball ID: " + type);
      
        string currentChatName = "Chat" + currentChatNumber;
        currentChat = historyChat.transform.Find(currentChatName).gameObject;
        currentChat.SetActive(true);
        bubble = currentChat.transform.Find("Bubble").gameObject;
        load = bubble.transform.Find("Load").gameObject;
        text = bubble.transform.Find("Text").gameObject;

        if (text.GetComponent<TextMeshPro>().enabled)
        {
            text.GetComponent<TextMeshPro>().enabled = false;
        }
        load.GetComponent<TextMeshPro>().enabled = true;
        ball.tag = "IDdBall";
        Thread.Sleep(3000);
        load.GetComponent<TextMeshPro>().enabled = false;
        text.GetComponent<TextMeshPro>().enabled = true;
        text.GetComponent<TextMeshPro>().text = $"This is a {type} ball!";
        currentChatNumber += 1;
        BallsIdentified.Add(ball.name);
    }
        
}
