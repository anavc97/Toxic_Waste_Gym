using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using CodeMonkey.Utils;
using UnityEditor.Scripting.Python;
using UnityEditor;

public class GameHandler : MonoBehaviour
{
    [SerializeField] private HealthBar healthBar;
    public float global_health;
    
    private void Start()
    {   
        FunctionPeriodic.Create(() => {
            if (global_health > 0){
                global_health -= 0.001f;
                healthBar.SetSize(global_health);            }

       
        }, 0.1f);
        
        ScoreScript.scoreValue = 0;
    }

    void Update()
    {   
        PythonRunner.RunFile($"{Application.dataPath}/Scripts/python_manager.py");

    }

    public void update_Score_Health(int score, float health)
    {   
        global_health = health;
        healthBar.SetSize(health);
        ScoreScript.scoreValue = score;
    }

}

