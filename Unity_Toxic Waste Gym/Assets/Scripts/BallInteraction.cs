using System.Collections;
using System.Collections.Generic;

using UnityEngine;
using TMPro;
using Unity.Collections.LowLevel.Unsafe;
using JetBrains.Annotations;

public class BallInteraction : MonoBehaviour
{
    public GameObject historyChat;
    public GameObject currentChat;
    private GameObject bubble;
    private GameObject load;
    private GameObject text;
    public GameObject[] allBalls;
    public List<string> BallsIdentified = new List<string>();
    public float Id_time;
    GameObject astroPlayer;

    private int currentChatNumber = 1;

    void Start()
    {
        historyChat = GameObject.Find("HistoryChat");
        allBalls = GameObject.FindGameObjectsWithTag("Ball");
        astroPlayer = GameObject.Find("astro");    
    }

    void Update()
    {
       Id_time = astroPlayer.GetComponent<ActionRenderingRobot>().Id_time;
    }

    public IEnumerator StartIdAnimation(GameObject ball, int index)
    {
        if(BallsIdentified.Contains(ball.name)) //Check if ball had already been identified before
        {
            yield return 0;
        }
        string type = ball.name.Split('_')[0];
        int nr= (currentChatNumber - 1) % 7 + 1;
        string currentChatName = "Chat" + nr;
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
        BallsIdentified.Add(ball.name);
        yield return new WaitForSeconds(Id_time);
        load.GetComponent<TextMeshPro>().enabled = false;
        text.GetComponent<TextMeshPro>().enabled = true;
        if(type == "green"){text.GetComponent<TextMeshPro>().color = new Color32(18,154,14,255);}
        else if(type == "red"){text.GetComponent<TextMeshPro>().color = new Color32(184,28,3,255);}
        else {text.GetComponent<TextMeshPro>().color = new Color32(240,154,4,255);}
        text.GetComponent<TextMeshPro>().text = $"Ball {index} is a {type} ball!";
        currentChatNumber += 1;
    }
    
    public bool checkPositionVacancy(Vector3 positionToCheck)
    {
        foreach(GameObject ball in allBalls)
        {
            if(ball != null && ball.GetComponent<SpriteRenderer>().enabled && ball.transform.position == positionToCheck)
            {
                return false; //Position occupied with a ball
            }
        }
        return true; //Position vacant
    }

    public GameObject findClosestBall(GameObject[] balls, Vector3 playerPosition, Vector2 playerOrientation, bool isHuman)
    {

        GameObject closestBall = null; //TODO: Instead of one object add a list and use as tie breaker for adjacent balls human orientation
        //Problem is if human is not facing any ball (no tie-break) so may actually have to force to be facing towards ball in order to pick it up
        float closestDistance = Mathf.Infinity;

        //Find the closest ball
        foreach (GameObject ball in balls)
        {   
            if (ball != null && ball.GetComponent<SpriteRenderer>().enabled)
            {
                float distance = Vector3.Distance(playerPosition, ball.transform.position);
                Vector3 facingTile = new Vector3(playerPosition.x + playerOrientation.x, playerPosition.y + playerOrientation.y, 0);
                if(isHuman && facingTile == ball.transform.position) //Force human to be facing ball
                {
                    return ball;
                }
                if (!isHuman && distance < closestDistance) //Might have to add tie break of facing ball if robot trying to identify a ball that is at same distance of another
                {
                    closestDistance = distance;
                    closestBall = ball;
                }
            }
        }
        return closestBall;
    }
        
}
