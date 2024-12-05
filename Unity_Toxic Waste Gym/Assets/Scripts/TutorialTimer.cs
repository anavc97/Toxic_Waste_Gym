using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class TutorialTimer : MonoBehaviour
{
    public static TutorialTimer Instance;
    public float tutorial_timer;
    private float start_timer;
    private void Awake()
    {
        tutorial_timer = 0;
        start_timer = Time.time;
        
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else if (Instance != this)
        {
            Destroy(gameObject);
        }
    }

    void Update()
    {
        if(SceneManager.GetActiveScene().name == "Tutorial" || SceneManager.GetActiveScene().name == "Tutorial 2" || SceneManager.GetActiveScene().name == "Tutorial 3")
        {
            Debug.Log(Time.time - start_timer);
            tutorial_timer = Time.time - start_timer;
        }

    }
}
