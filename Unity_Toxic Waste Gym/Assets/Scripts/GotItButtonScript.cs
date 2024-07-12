using UnityEngine;
using UnityEngine.SceneManagement;

public class GotItButtonScript : MonoBehaviour
{
    public GameObject canvas;
    public Timer timerScript;
    void Start()
    {   
        //timerScript = GameObject.Find("Timer").GetComponent<Timer>();
        //timerScript.timeIsRunning = false;
        canvas = GameObject.FindWithTag("Canvas");
        //canvas.GetComponent<Canvas>().enabled = true;
        //Time.timeScale = 0f;
        Debug.Log("Time: " + Time.timeScale);
    }

    public void ResumeGame()
    {
        // Resume the game
        Time.timeScale = 1f;
        timerScript.timeIsRunning = true;
        // Hide the pause panel
        canvas.GetComponent<Canvas>().enabled = true;
    }
}
