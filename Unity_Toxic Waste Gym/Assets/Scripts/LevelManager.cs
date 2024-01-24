using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEditor.Scripting.Python;
using UnityEditor;

public class LevelManager : MonoBehaviour
{
    // Start is called before the first frame update
    public string sceneName;
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void changeScene(string sceneName)
    {
        PythonRunner.RunFile($"{Application.dataPath}/Scripts/StartGame.py");
        SceneManager.LoadScene(sceneName);
    }
}
