using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
public class ScoreScript: MonoBehaviour
{
    public static int scoreValue = 0;
    public TMP_Text score;

    void Start()
    {
        score.text = "Score: 0";
    }

    void Update()
    {
        score.text = "Score: " + scoreValue;
    }
}
