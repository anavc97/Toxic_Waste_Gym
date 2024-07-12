using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class GridLimits : MonoBehaviour
{
    public List<Vector3> gridPositions;
    public List<Vector3> gridPosAvailable;

    private static readonly List<(int, int, int, int)> wallRanges = new List<(int, int, int, int)>
    {
        (3, 9, 8, 9),
        (6, 8, 8, 8),
        (6, 7, 8, 7),
        (6, 6, 8, 6),
        (6, 5, 11, 5)
    };
    
    void Start()
    {
        Scene currentScene = SceneManager.GetActiveScene();
		if(currentScene.name == "level_zero"){defineGridLevelZero();}
        else if(currentScene.name == "level_one"){defineGridLevelOne();}
        else if(currentScene.name == "level_two"){defineGridLevelTwo();}
        else if(currentScene.name == "level_three"){defineGridLevelThree();}

        gridPosAvailable = new List<Vector3>();

        for (int x = 0; x <= 14; x++)
        {
            for (int y = 0; y <= 14; y++)
            {
                Vector3 position = new Vector3(x, y, 0);
                if (!gridPositions.Contains(position) && position != new Vector3(7,14,0))
                {
                    gridPosAvailable.Add(position);
                }
            }
        }
    }

    void defineGridLevelZero()
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
                else if (y == 14 && (x == 6 || x == 7 || x == 8)) //Door wall
                {;}
                else if ((x == 0 && (y >= 1 && y < 14)) || (x == 14 && (y >= 1 && y < 14))) //Outside horizontal
                    gridPositions.Add(new Vector3(x, y,0));
                else if ((y == 0 && (x >= 1 && x < 14)) || (y == 14 && (x >= 1 && x < 6)) || (y == 14 && (x >= 9 && x < 14)))//Outside vertical
                    gridPositions.Add(new Vector3(x, y,0));
            }
        }

        foreach ((int startX, int startY, int endX, int endY) in wallRanges)
        {
            for (int x = startX; x <= endX; x++)
            {
                for (int y = startY; y <= endY; y++)
                {
                    gridPositions.Add(new Vector3(x, y, 0));
                }
            }
        }
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
                else if (y == 14 && (x == 6 || x == 7 || x == 8)) //Door wall
                {;}
                else if ((x == 0 && (y >= 1 && y < 14)) || (x == 14 && (y >= 1 && y < 14))) //Outside horizontal
                    gridPositions.Add(new Vector3(x, y,0));
                else if ((y == 0 && (x >= 1 && x < 14)) || (y == 14 && (x >= 1 && x < 6)) || (y == 14 && (x >= 9 && x < 14)))//Outside vertical
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
                else if (y == 14 && (x == 6 || x == 7 || x == 8)) //Door wall
                {;}
                else if ((x == 0 && (y >= 1 && y < 14)) || (x == 14 && (y >= 1 && y < 14))) //Outside horizontal
                    gridPositions.Add(new Vector3(x, y,0));
                else if ((y == 0 && (x >= 1 && x < 14)) || (y == 14 && (x >= 1 && x < 6)) || (y == 14 && (x >= 9 && x < 14)))//Outside vertical
                    gridPositions.Add(new Vector3(x, y,0));
                else if ((y == 1 && (x == 1 || x == 2)) || (y == 1 && (x >= 7 && x <= 10)) || 
                            (y == 1 && (x == 12 || x == 13)) || (y == 2 && (x == 9 || x == 10)) ||
                            (y == 2 && (x == 12 || x == 13)) || (y == 3 && (x == 9 || x == 10)) ||
                            (y == 3 && (x == 4 || x == 5)) || (y == 4 && (x >= 2 && x <= 7)) ||
                            (y == 5 && (x >= 2 && x <= 5)) || (y == 5 && (x >= 9 && x <= 11)) ||
                            (y == 6 && (x >= 2 && x <= 5)) || (y == 6 && (x >= 7 && x <= 11)) ||
                            (y == 7 && (x == 2 || x == 3)) || (y == 8 && x == 2) ||
                            (y == 9 && (x >= 5 && x <= 11)) || (y == 10 && (x >= 5 && x <= 11)) ||
                            (y == 10 && (x >= 1 && x <= 3)) || ((x == 6 || x==10) && y == 11) || 
                            (y == 12 && (x >= 2 && x <= 4)) || (x==9 && (y==11 || y==12)))  
                    gridPositions.Add(new Vector3(x, y,0));
            }
        }
    }

    void defineGridLevelThree()
    {
        string grid = 
            "XXXXXX   XXXXXX" +
            "X    X   X    X" +
            "X XX X   X XX X" +
            "X X         X X" +
            "X XX   X   XX X" +
            "X     XXX     X" +
            "X    XXXXX    X" +
            "X   XXXXXXX   X" +
            "XX     X     XX" +
            "XXXXX  X  XXXXX" +
            "X   X  X  X   X" +
            "XX           XX" +
            "XXXXX     XXXXX" +
            "X             X" +
            "XXXXXXXXXXXXXXX";

        gridPositions = new List<Vector3>();

        int width = 15; // Width of the grid
        int height = 15; // Height of the grid

        // Iterate over each character in the string
        for (int y = height - 1; y >= 0; y--) // Starting from y = 14 to y = 0
        {
            for (int x = 0; x < width; x++)
            {
                int index = (height - y - 1) * width + x; // Calculate the index in the string
                char character = grid[index]; // Get the character at the index

                if (character == 'X')
                {
                    // Add the position to the list
                    gridPositions.Add(new Vector3(x, y, 0));
                }
            }
        }
    }

}
