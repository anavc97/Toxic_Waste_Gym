using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using TMPro; // If using TextMeshPro

public class ChangeTrustValue : MonoBehaviour
{
    [SerializeField] public Slider slider;
    [SerializeField] public TextMeshProUGUI valueText; // If using TextMeshPro
    public LogManager logger;
    public GameHandler gameHandler;
    // Start is called before the first frame update
    void Start()
    {
        //if(SceneManager.GetActiveScene().name == "level_three"){gameObject.SetActive(false);}  
        logger = GameObject.Find("LogManager").GetComponent<LogManager>();
        gameHandler = GameObject.Find("GameHandler").GetComponent<GameHandler>();
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    public void UpdateValueText()
    {
        valueText.text = slider.value.ToString("0");
    }

    public void UpdateTrust()
    {
        int ind = gameHandler.sceneList.IndexOf(SceneManager.GetActiveScene().name);
        Debug.Log("List index: " + ind);
        Debug.Log("trust value " + ind + "  before: " + logger.trustValueList[ind]);
        logger.trustValueList[ind] = slider.value;
        Debug.Log("trust value" + ind + " after: " + logger.trustValueList[ind]);
        gameHandler.trustSubmitted = true;
    }

}
