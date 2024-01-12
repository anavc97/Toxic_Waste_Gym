using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using CodeMonkey.Utils;

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
        
        ScoreScript.scoreValue = 20;
    }


}
