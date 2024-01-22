using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using CodeMonkey.Utils;
using UnityEditor.Scripting.Python;
using UnityEditor;

public class GameHandler : MonoBehaviour
{
    [SerializeField] private HealthBar healthBar;
    private void Start()
    {   
        float health = 1f;
        FunctionPeriodic.Create(() => {
            if (health > 0){
                health -= 0.001f;
                healthBar.SetSize(health);            }

       
        }, 0.1f);
        
        ScoreScript.scoreValue = 0;
    }

    void Update()
    {
        PythonRunner.RunFile($"{Application.dataPath}/Scripts/python_manager.py");

    }

    public void update_Score_Health(float score, float health)
    {   
        Debug.Log("Score and Health:");
        Debug.Log(score);
        Debug.Log(health)
    }

}

