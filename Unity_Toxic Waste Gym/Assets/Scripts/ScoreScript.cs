using UnityEngine;
using TMPro;
public class ScoreScript: MonoBehaviour
{
    public float scoreValue = 0;
    public TMP_Text score;

    void Start()
    {
        score.text = "Score\n0";
    }

    void Update()
    {
        score.text = "Score\n" + scoreValue;
    }
}
