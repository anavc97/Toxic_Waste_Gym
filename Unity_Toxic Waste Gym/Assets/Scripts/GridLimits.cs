using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class GridLimits : MonoBehaviour
{
    public List<Vector3> gridPositions;
    public List<Vector3> gridPosAvailable;
    
    void Start()
    {
        Scene currentScene = SceneManager.GetActiveScene();
		if(currentScene.name == "level_one"){defineGridLevelOne();}
        else if(currentScene.name == "level_two"){defineGridLevelTwo();}
    }

    void defineGridLevelOne()
    {
        gridPositions = new List<Vector3>();

        // Add positions to the list using nested loops
        for (int x = 0; x <= 14; x++)
        {
            for (int y = 0; y <= 14; y++)
            {
                // Add positions according to the specified patterns
                if ((y == 0 && (x == 0 || x == 14)) || (y == 14 && (x == 0 || x == 14))) //Corners
                    gridPositions.Add(new Vector3(x, y,0));
                else if (y == 14 && x == 7) //Door wall
                    gridPositions.Add(new Vector3(x, y+1,0));
                else if ((x == 0 && (y >= 1 && y < 14)) || (x == 14 && (y >= 1 && y < 14))) //Outside horizontal
                    gridPositions.Add(new Vector3(x, y,0));
                else if ((y == 0 && (x >= 1 && x < 14)) || (y == 14 && (x >= 1 && x < 7)) || (y == 14 && (x >= 8 && x < 14)))//Outside vertical
                    gridPositions.Add(new Vector3(x, y,0));
                else if ((x == 1 && y == 2) || (x == 1 && y == 3) || (x == 2 && (y == 2 || y == 3)) ||
                            ((x >= 5 && x <= 9) && y == 2) || ((x >= 12 && x <= 13) && y == 2) ||
                            ((x >= 6 && x <= 8) && y == 3) || ((x >= 6 && x <= 8) && y == 4) || ((x >= 10 && x <= 11) && y == 4) ||
                            (x == 7 && (y == 5 || y == 6 || y == 7)) || ((x >= 10 && x <= 11) && (y == 5 || y == 6 || y == 7)) ||
                            (x == 4 && y == 5) || ((x >= 2 && x <= 4) && y == 6) || ((x == 9 || x == 12) && y == 6) || 
                            (x == 5 && y == 8) || ((x >= 10 && x <= 11) && y == 8) || ((x >= 1 && x <= 5) && y == 9) ||
                            (x == 7 && y == 9) || ((x >= 1 && x <= 5) && y == 10) || ((x >= 8 && x <= 11) && y == 11) ||
                            ((x >= 2 && x <= 6) && y == 12) || ((x >= 10 && x <= 11) && y == 12))
                    gridPositions.Add(new Vector3(x, y,0));
            }
        }

        gridPosAvailable = new List<Vector3>();

        for (int x = 0; x <= 14; x++)
        {
            for (int y = 0; y <= 14; y++)
            {
                Vector3 position = new Vector3(x, y, 0);
                if (!gridPositions.Contains(position))
                {
                    gridPosAvailable.Add(position);
                }
            }
        }
    }
    
    void defineGridLevelTwo()
    {
        gridPositions = new List<Vector3>();

        // Add positions to the list using nested loops
        for (int x = 0; x <= 14; x++)
        {
            for (int y = 0; y <= 14; y++)
            {
                // Add positions according to the specified patterns
                if ((y == 0 && (x == 0 || x == 14)) || (y == 14 && (x == 0 || x == 14))) //Corners
                    gridPositions.Add(new Vector3(x, y,0));
                else if (y == 14 && x == 7) //Door wall
                    gridPositions.Add(new Vector3(x, y+1,0));
                else if ((x == 0 && (y >= 1 && y < 14)) || (x == 14 && (y >= 1 && y < 14))) //Outside horizontal
                    gridPositions.Add(new Vector3(x, y,0));
                else if ((y == 0 && (x >= 1 && x < 14)) || (y == 14 && (x >= 1 && x < 7)) || (y == 14 && (x >= 8 && x < 14)))//Outside vertical
                    gridPositions.Add(new Vector3(x, y,0));
                else if ((y == 1 && (x == 1 || x == 2)) || (y == 1 && (x >= 7 && x <= 10)) || 
                            (y == 1 && (x == 12 || x == 13)) || (y == 2 && (x == 9 || x == 10)) ||
                            (y == 2 && (x == 12 || x == 13)) || (y == 3 && (x == 9 || x == 10)) ||
                            (y == 3 && (x == 4 || x == 5)) || (y == 4 && (x >= 2 && x <= 7)) ||
                            (y == 5 && (x >= 2 && x <= 5)) || (y == 5 && (x >= 9 && x <= 11)) ||
                            (y == 6 && (x >= 2 && x <= 5)) || (y == 6 && (x >= 7 && x <= 11)) ||
                            (y == 7 && (x == 2 || x == 3)) || (y == 8 && x == 2) || (y == 11 && x == 5) ||
                            (y == 9 && (x >= 5 && x <= 11)) || (y == 10 && (x >= 5 && x <= 11)) ||
                            (y == 10 && (x >= 1 && x <= 3)) || (x == 6 && (y >= 11 && y <= 13)) ||
                            (y == 11 && x == 11) || (y == 13 & x == 5) || (y == 12 && x == 11) ||
                            (y == 12 && (x >= 2 && x <= 5)) || (x == 8 && (y == 12 || y == 13)))  
                    gridPositions.Add(new Vector3(x, y,0));
            }
        }

        gridPosAvailable = new List<Vector3>();

        for (int x = 0; x <= 14; x++)
        {
            for (int y = 0; y <= 14; y++)
            {
                Vector3 position = new Vector3(x, y, 0);
                if (!gridPositions.Contains(position))
                {
                    gridPosAvailable.Add(position);
                }
            }
        }
    }
}
