using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using TMPro; // If using TextMeshPro

public class UpdateManipulationCheck : MonoBehaviour
{

    [SerializeField] public Slider slider;
    [SerializeField] public TextMeshProUGUI valueText; // If using TextMeshPro
    public LogManager logger;
    public GameHandler gameHandler;

    private List<string> TextList= new List<string>{"Very Badly","Badly", "Okay", "Well", "Very Well", "I'm not sure"};
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
        int ind = (int)slider.value-1;
        valueText.text = TextList[ind];
    }

    public void UpdateManCheck()
    {
        int ind = gameHandler.sceneList.IndexOf(SceneManager.GetActiveScene().name);
        Debug.Log("List index: " + ind);
        Debug.Log("man value " + ind + "before: " + logger.manCheckList[ind]);
        logger.manCheckList[ind] = slider.value;
        Debug.Log("man value" + ind + " after: " + logger.manCheckList[ind]);
    }
}
