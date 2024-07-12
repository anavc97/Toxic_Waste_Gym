using UnityEngine;
using TMPro;
using UnityEngine.SceneManagement;
public class ScoreScript: MonoBehaviour
{
    public float scoreValue = 0;
    public float globalScore = 0;
    public TMP_Text score;

    void Start()
    {
        score.text = "Score\n0";
        DontDestroyOnLoad(gameObject);
    }

    void Update()
    {
        score.text = "Score\n" + scoreValue;
    }
}
