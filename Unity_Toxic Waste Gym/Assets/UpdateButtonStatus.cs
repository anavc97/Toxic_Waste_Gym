using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Video;

public class UpdateButtonStatus : MonoBehaviour
{
    public GameObject button;
    public VideoPlayer m_VideoPlayer;
    
    // Start is called before the first frame update
    void Start()
    {
        m_VideoPlayer = GetComponent<VideoPlayer>();
        button.SetActive(false);
    }

    // Update is called once per frame
    void Update()
    {   
        UnityEngine.Debug.Log("HELLO");
        UnityEngine.Debug.Log(button.activeSelf);
        if (!button.activeSelf)
        {
            if ( m_VideoPlayer.frame > 0 && (m_VideoPlayer.isPlaying == false))
                {
                    button.SetActive(true);
                    Debug.Log("Finished!!!");
                }
        }
 
    }
}
