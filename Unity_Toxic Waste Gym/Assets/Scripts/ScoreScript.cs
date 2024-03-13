using UnityEngine;
using TMPro;
public class ScoreScript: MonoBehaviour
{
    public static float scoreValue = 0;
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
