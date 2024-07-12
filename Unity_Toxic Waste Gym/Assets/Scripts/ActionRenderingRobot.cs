using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using Newtonsoft.Json;
using System.Linq;
using System;
using System.Diagnostics.Tracing;
using Unity.VisualScripting;
//using System.Numerics;

public class ActionRenderingRobot : MonoBehaviour
{
    public class AStarPathfinding
    {   
      public static List<Vector3> FindPath(List<Vector3> gridPositions, Vector3 origin, Vector3 target)
      {
          // Convert grid positions to grid cells
          Dictionary<Vector3, bool> gridCells = new Dictionary<Vector3, bool>();
          foreach (Vector3 position in gridPositions)
          {
              gridCells[position] = true;
          }

          List<Vector3> path = new List<Vector3>();

          // Initialize start and goal cells
          Vector3 startCell = GetClosestCell(origin, gridCells);
          Vector3 goalCell = GetClosestCell(target, gridCells);

          if (startCell == Vector3.zero || goalCell == Vector3.zero)
          {
              Debug.Log("Unable to find path. Invalid start or goal position.");
              return path;
          }

          List<Vector3> openList = new List<Vector3>();
          List<Vector3> closedList = new List<Vector3>();
          Dictionary<Vector3, Vector3> parentMap = new Dictionary<Vector3, Vector3>();

          openList.Add(startCell);

          while (openList.Count > 0)
          {
              Vector3 currentCell = GetLowestCostCell(openList, startCell, goalCell);
              openList.Remove(currentCell);
              closedList.Add(currentCell);

              if (currentCell == goalCell)
              {
                  // Path found, reconstruct and return the path
                  path = ConstructPath(startCell, goalCell, parentMap);
                  break;
              }

              foreach (Vector3 neighbor in GetNeighbors(currentCell, gridCells))
              {
                  if (closedList.Contains(neighbor))
                      continue;

                  if (!openList.Contains(neighbor))
                  {
                      openList.Add(neighbor);
                      parentMap[neighbor] = currentCell;
                  }
              }
          }

          return path;
      }

      private static Vector3 GetClosestCell(Vector3 position, Dictionary<Vector3, bool> gridCells)
      {
          float minDistance = float.MaxValue;
          Vector3 closestCell = Vector3.zero;

          foreach (Vector3 cell in gridCells.Keys)
          {
              float distance = Vector3.Distance(position, cell);
              if (distance < minDistance)
              {
                  minDistance = distance;
                  closestCell = cell;
              }
          }

          return closestCell;
      }

      private static Vector3 GetLowestCostCell(List<Vector3> openList, Vector3 start, Vector3 goal)
      {
          Vector3 lowestCostCell = openList[0];
          float lowestCost = Vector3.Distance(start, lowestCostCell) + Vector3.Distance(lowestCostCell, goal);

          foreach (Vector3 cell in openList)
          {
              float cost = Vector3.Distance(start, cell) + Vector3.Distance(cell, goal);
              if (cost < lowestCost)
              {
                  lowestCost = cost;
                  lowestCostCell = cell;
              }
          }

          return lowestCostCell;
      }

      private static List<Vector3> ConstructPath(Vector3 start, Vector3 goal, Dictionary<Vector3, Vector3> parentMap)
      {
          List<Vector3> path = new List<Vector3>();
          Vector3 current = goal;

          while (current != start)
          {
              path.Add(current);
              current = parentMap[current];
          }

          path.Add(start);
          path.Reverse();

          return path;
      }

      private static IEnumerable<Vector3> GetNeighbors(Vector3 cell, Dictionary<Vector3, bool> gridCells)
      {
          Vector2Int[] directions = { Vector2Int.up, Vector2Int.down, Vector2Int.left, Vector2Int.right };

          foreach (Vector2Int dir in directions)
          {
              Vector3 neighborPosition = cell + new Vector3(dir.x, dir.y, 0);

              if (gridCells.ContainsKey(neighborPosition))
              {
                  yield return neighborPosition;
              }
          }
      }
    }
    public float movementSpeed;
    private Animator animator;
    public Vector2 astroOrientation;
    public Dictionary<string, bool> BallsIdd;

    public BallInteraction ballInteraction;
    public GameObject humanPlayer;
    private List<Vector3> walls; 
    private List<Vector3> floor; 
    public List<Vector3> pos_around = new List<Vector3>();
    public Dictionary<int, int> oppositeStepActions = new Dictionary<int, int>(){{-1,-1},{0,1},{1,0},{2,3},{3,2}};
    public float pos_x;
    public float pos_y;
    public bool gameOverRobot;
    public Vector3 astroStation;
    public bool error = false;
    public float Id_time;
    GameObject[] allBalls;

    private GameObject targetBall;
    
    void Awake()
    {
      animator = GetComponent<Animator>();

    }

    // Start is called before the first frame update
    void Start()
    {

      astroOrientation = new Vector2(0,-1);
      humanPlayer = GameObject.Find("human");
      ballInteraction = GameObject.Find("red_1").GetComponent<BallInteraction>();
      Scene currentScene = SceneManager.GetActiveScene();
		  if(currentScene.name == "level_one"){astroStation = new Vector3(7,8,0);}
      else if(currentScene.name == "level_two"){astroStation = new Vector3(7,7,0);}
      else if(currentScene.name == "level_three"){astroStation = new Vector3(6,12,0);}
      else if(currentScene.name == "level_zero"){astroStation = new Vector3(7,10,0);}
      walls = GameObject.Find("Grid").GetComponent<GridLimits>().gridPositions;
      floor = GameObject.Find("Grid").GetComponent<GridLimits>().gridPosAvailable;
      gameOverRobot = false;
      allBalls = GameObject.FindGameObjectsWithTag("Ball");
      if(SceneManager.GetActiveScene().name == "level_three" || SceneManager.GetActiveScene().name == "level_zero"){StartCoroutine(AstroAutomatic());}
      else{
        //StartCoroutine(AstroAutomatic());
        StartCoroutine(AstroBad());
        //StartCoroutine(AstroBadSimple());
      }
    }

    private IEnumerator ActivateError()
    {
      while(!gameOverRobot){

        int a = UnityEngine.Random.Range(0, 10);
        if (a > 3){  
          error = true;
          yield return new WaitForSeconds(7f);
          error = false;
        }
        
        yield return new WaitForSeconds(13f);
      }
    }

    void Update()
    {
      if(gameOverRobot){this.enabled = false;Debug.Log("GAME OVER ROBOT");}
    }
    

    IEnumerator AstroAutomatic()
    { 
      Id_time = 4f;
      yield return new WaitForSeconds(1f);
      GameObject[] balls = GameObject.FindGameObjectsWithTag("Ball"); 
      float distanceToHuman = Mathf.Infinity;
      bool targetLocked = false;
      Debug.Log("Started astro automatic");
      while (balls.Length != 0)
      { 
        float closestDistance = Mathf.Infinity;
        balls = GameObject.FindGameObjectsWithTag("Ball");
        if(balls.Length == 0){
          break;}
        //GameObject randomBall = balls[UnityEngine.Random.Range(0, balls.Length)];
        targetBall = null;

        while (closestDistance > Math.Round(Mathf.Sqrt(2), 2) + 0.01 || closestDistance <= 1) //Mathf.Sqrt(2))
        { 

          balls = GameObject.FindGameObjectsWithTag("Ball");
          if(balls.Length == 0){break;}
          if(!targetLocked)
          { 
            // Choosing between closest ball to Astro and closest ball to Human
            (GameObject closestBallAstro, float DistanceBallAstro) = FindClosestBall(balls, gameObject);
            (GameObject closestBallHuman, float DistanceBallHuman) = FindClosestBall(balls, humanPlayer);

            if (DistanceBallAstro <= DistanceBallHuman)
            {
              targetBall = closestBallAstro;
            }
            else
            {
              targetBall = closestBallHuman;
            }
            
            targetLocked = true;
          }
          //Move towards human if human is holding a ball
          while(humanHoldingBall(allBalls))
          { 
            if(distanceToHuman <= Math.Round(Mathf.Sqrt(2), 2) + 0.01)
            {
              yield return new WaitForSeconds(0.4f);
              distanceToHuman = Vector3.Distance(transform.position, humanPlayer.transform.position);
              continue;
            }
            distanceToHuman = Vector3.Distance(transform.position, humanPlayer.transform.position);
            ObtainNextAction(humanPlayer.transform.position);
            yield return new WaitForSeconds(0.4f);
          }
          
          //Move towards target ball if human isn't holding a ball
          Vector3 newTarget = targetBall.transform.position + new Vector3(1.0f,1.0f,0);
          ObtainNextAction(newTarget);

          closestDistance = Vector3.Distance(transform.position, targetBall.transform.position);
          //moveOrRotateRobot(next_step, new Vector2(0,-1));
          if(targetBall.tag == "CollectedBall"){break;}
          yield return new WaitForSeconds(0.4f);
        }

        //Identify ball
        balls = GameObject.FindGameObjectsWithTag("Ball");
        if(balls.Length == 0){
          break;}
        Coroutine co = StartCoroutine(ballInteraction.StartIdAnimation(targetBall,Array.IndexOf(allBalls, targetBall), false));
        targetLocked = false;
        closestDistance = Vector3.Distance(transform.position, targetBall.transform.position);
        if(closestDistance > 2){StopCoroutine(co);ballInteraction.CancelIdAnimation();continue;}
        
        yield return new WaitForSeconds(Id_time/3);
        
        closestDistance = Vector3.Distance(transform.position, targetBall.transform.position);
        if(closestDistance > 2){StopCoroutine(co);ballInteraction.CancelIdAnimation();continue;}
        
        yield return new WaitForSeconds(Id_time/3);

        closestDistance = Vector3.Distance(transform.position, targetBall.transform.position);
        if(closestDistance > 2){StopCoroutine(co);ballInteraction.CancelIdAnimation();continue;}

        yield return new WaitForSeconds(Id_time/3);
      }

      //When all balls identified robot still follows human when he's holding a ball
      balls = GameObject.FindGameObjectsWithTag("IDdBall");
      while(balls.Length != 0)
      {
        distanceToHuman = Vector3.Distance(transform.position, humanPlayer.transform.position);
        if(humanHoldingBall(balls) && distanceToHuman > Math.Round(Mathf.Sqrt(2), 2) + 0.01) //Move towards human
        {
          ObtainNextAction(humanPlayer.transform.position);
        }
        else if(!humanHoldingBall(balls)) //Move towards ball closest to human
        {
          Vector3 closestBall2Human = GetClosestBallPositionToHuman();
          Vector3 NewTarget = closestBall2Human + new Vector3(1.0f,1.0f,0);
          ObtainNextAction(NewTarget);
          //ObtainNextAction(astroStation);
        }
        yield return new WaitForSeconds(0.4f);
        balls = GameObject.FindGameObjectsWithTag("IDdBall");
      }

      //Return to base at the end
      while(transform.position != astroStation)
      {
        ObtainNextAction(astroStation);
        yield return new WaitForSeconds(0.4f);
      }
    } 
    
    IEnumerator AstroBad()
    { 
      Id_time = 8f;
      StartCoroutine(ActivateError());
      yield return new WaitForSeconds(1f);
      GameObject[] balls = GameObject.FindGameObjectsWithTag("Ball");
      float distanceToHuman = Mathf.Infinity;
      Debug.Log("Started astro bad");
      GameObject[] allGoodBalls = allBalls.Where(obj => obj.tag != "CollectedBall").ToArray();
      bool HHoldBall = humanHoldingBall(allBalls);
      if(error){HHoldBall = !humanHoldingBall(allBalls);}

      while (balls.Length != 0)
      { 
        allGoodBalls = allBalls.Where(obj => obj.tag != "CollectedBall").ToArray();
        float closestDistance = Mathf.Infinity;
        balls = GameObject.FindGameObjectsWithTag("Ball");
        if(balls.Length == 0){break;}
        HHoldBall = humanHoldingBall(allBalls);
        if(error){HHoldBall = !humanHoldingBall(allBalls);}
        GameObject randomBall = allGoodBalls[UnityEngine.Random.Range(0, balls.Length)];

        if(Vector3.Distance(transform.position, randomBall.transform.position)<=2){randomBall = allGoodBalls[UnityEngine.Random.Range(0, balls.Length)];}
        GameObject[] identifiedBalls = GameObject.FindGameObjectsWithTag("IDdBall");
        
        while (closestDistance > Mathf.Sqrt(2)) //Mathf.Sqrt(2))
        { 
          //Move towards human if human is holding a ball - with probability of failing for 5 seconds
          while(HHoldBall)
          {          
            HHoldBall = humanHoldingBall(allBalls);
            if(error){HHoldBall = !humanHoldingBall(allBalls);}
            if(distanceToHuman <= Mathf.Sqrt(2))
            {
              yield return new WaitForSeconds(0.4f);
              distanceToHuman = Vector3.Distance(transform.position, humanPlayer.transform.position);
              continue;
            }
            distanceToHuman = Vector3.Distance(transform.position, humanPlayer.transform.position);
            ObtainNextAction(humanPlayer.transform.position);
            yield return new WaitForSeconds(0.4f);
          }
          
          //Move towards closest ball if human isn't holding a ball
          balls = GameObject.FindGameObjectsWithTag("Ball");
          if(balls.Length ==0){break;}

          //Go to random ball
          Vector3 newTarget = randomBall.transform.position + new Vector3(1.0f,0,0);
          if(error){newTarget = newTarget + new Vector3(-3.0f,+2.0f,0);}
          ObtainNextAction(newTarget);
          closestDistance = Vector3.Distance(transform.position, randomBall.transform.position);
          //moveOrRotateRobot(next_step, new Vector2(0,-1));
          if(randomBall.tag == "CollectedBall"){break;}
          yield return new WaitForSeconds(0.4f);
        }
        
        //Identify random ball
        balls = GameObject.FindGameObjectsWithTag("Ball");
        int i = UnityEngine.Random.Range(0, allBalls.Length);
        GameObject wrongBall = allBalls[i];
        if(balls.Length == 0){break;}
        Coroutine co = StartCoroutine(ballInteraction.StartIdAnimation(wrongBall,System.Array.IndexOf(allBalls, randomBall), wrongBall!=randomBall));

        closestDistance = Vector3.Distance(transform.position, randomBall.transform.position);
        if(closestDistance > 2){StopCoroutine(co);ballInteraction.CancelIdAnimation();continue;}
        
        yield return new WaitForSeconds(Id_time/3);
        
        closestDistance = Vector3.Distance(transform.position, randomBall.transform.position);
        if(closestDistance > 2){StopCoroutine(co);ballInteraction.CancelIdAnimation();continue;}
        
        yield return new WaitForSeconds(Id_time/3);

        closestDistance = Vector3.Distance(transform.position, randomBall.transform.position);
        if(closestDistance > 2){StopCoroutine(co);ballInteraction.CancelIdAnimation();continue;}

        yield return new WaitForSeconds(Id_time/3);
      }

      //When all balls identified robot still follows human when he's holding a ball
      balls = GameObject.FindGameObjectsWithTag("IDdBall");
      allGoodBalls = allBalls.Where(obj => obj.tag != "CollectedBall").ToArray();
      StartCoroutine(ActivateError());

      while(balls.Length != 0)
      {      
        HHoldBall = humanHoldingBall(allBalls);
        if(error){HHoldBall = !humanHoldingBall(allBalls);}
        distanceToHuman = Vector3.Distance(transform.position, humanPlayer.transform.position);
        if(HHoldBall && distanceToHuman > Mathf.Sqrt(2)) //Move towards human
        {
          ObtainNextAction(humanPlayer.transform.position);
        }
        else if(!HHoldBall && transform.position != astroStation) //Move towards station
        {
          ObtainNextAction(astroStation);
        }
        yield return new WaitForSeconds(0.4f);
        balls = GameObject.FindGameObjectsWithTag("IDdBall");
      }
      
      //Return to base at the end
      while(transform.position != astroStation)
      {
        ObtainNextAction(astroStation);
        yield return new WaitForSeconds(0.4f);
      }
    }  

    // Only id balls wrong
    IEnumerator AstroBadSimple()
    { 
      Id_time = 5f;
      yield return new WaitForSeconds(1f);
      GameObject[] balls = GameObject.FindGameObjectsWithTag("Ball"); 
      float distanceToHuman = Mathf.Infinity;
      bool targetLocked = false;
      GameObject targetBall = null;
      Debug.Log("Started astro automatic");
      while (balls.Length != 0)
      { 
        float closestDistance = Mathf.Infinity;
        balls = GameObject.FindGameObjectsWithTag("Ball");
        if(balls.Length == 0){
          break;}
        //GameObject randomBall = balls[UnityEngine.Random.Range(0, balls.Length)];
        
        GameObject lastTargetBall = targetBall;
        
        while (closestDistance > Math.Round(Mathf.Sqrt(2), 2) + 0.01) //Mathf.Sqrt(2))
        { 
          
          if(!targetLocked)
          { 
            // Choosing between closest ball to Astro and closest ball to Human
            (GameObject closestBallAstro, float DistanceBallAstro) = FindClosestBall(balls, gameObject);
            (GameObject closestBallHuman, float DistanceBallHuman) = FindClosestBall(balls, humanPlayer);

            if (DistanceBallAstro <= DistanceBallHuman)
            {
              targetBall = closestBallAstro;
              if(lastTargetBall==targetBall){
                targetBall = closestBallHuman;} // so its not stuck on the same ball forever
            }
            else
            {
              targetBall = closestBallHuman;
              if(lastTargetBall==targetBall){
                targetBall = closestBallAstro;} // so its not stuck on the same ball forever
            }
            
            targetLocked = true;
          }
          //Move towards human if human is holding a ball
          while(humanHoldingBall(allBalls))
          { 
            if(distanceToHuman <= Math.Round(Mathf.Sqrt(2), 2) + 0.01)
            {
              yield return new WaitForSeconds(0.4f);
              distanceToHuman = Vector3.Distance(transform.position, humanPlayer.transform.position);
              continue;
            }
            distanceToHuman = Vector3.Distance(transform.position, humanPlayer.transform.position);
            ObtainNextAction(humanPlayer.transform.position);
            yield return new WaitForSeconds(0.4f);
          }
          
          //Move towards target ball if human isn't holding a ball
          Vector3 newTarget = targetBall.transform.position + new Vector3(1.0f,0,0);
          ObtainNextAction(newTarget);

          closestDistance = Vector3.Distance(transform.position, targetBall.transform.position);
          //moveOrRotateRobot(next_step, new Vector2(0,-1));
          yield return new WaitForSeconds(0.4f);
        }
        

        lastTargetBall = targetBall;
        if(targetBall!=null){//Identify ball
        balls = GameObject.FindGameObjectsWithTag("Ball");
        int i = UnityEngine.Random.Range(0, allBalls.Length);
        GameObject wrongBall = allBalls[i];
        if(balls.Length == 0){break;}
        StartCoroutine(ballInteraction.StartIdAnimation(wrongBall,System.Array.IndexOf(allBalls, targetBall), wrongBall!=targetBall));}
        
        targetLocked = false;
        
        yield return new WaitForSeconds(Id_time);
      }

      //When all balls identified robot still follows human when he's holding a ball
      balls = GameObject.FindGameObjectsWithTag("IDdBall");
      while(balls.Length != 0)
      {
        distanceToHuman = Vector3.Distance(transform.position, humanPlayer.transform.position);
        if(humanHoldingBall(balls) && distanceToHuman > Math.Round(Mathf.Sqrt(2), 2) + 0.01) //Move towards human
        {
          ObtainNextAction(humanPlayer.transform.position);
        }
        else if(!humanHoldingBall(balls)) //Move towards ball closest to human
        {
          Vector3 closestBall2Human = GetClosestBallPositionToHuman();
          ObtainNextAction(closestBall2Human);
          //ObtainNextAction(astroStation);
        }
        yield return new WaitForSeconds(0.4f);
        balls = GameObject.FindGameObjectsWithTag("IDdBall");
      }

      //Return to base at the end
      while(transform.position != astroStation)
      {
        ObtainNextAction(astroStation);
        yield return new WaitForSeconds(0.4f);
      }
    } 

    //Obtain next step action according to target destination (move towards human or ball)
    public void ObtainNextAction(Vector3 targetPosition)
    {
      pos_x = transform.position.x;
      pos_y = transform.position.y;
      List<Vector3>  moves = AStarPathfinding.FindPath(floor,transform.position,targetPosition);
      if (moves.Count > 1){      
        setNextOrientation(moves[1]);
        moveOrRotateRobot(moves[1], astroOrientation);}
    }

    Vector3 GetClosestBallPositionToHuman()
    {
        // Initialize variables to track the closest ball and its distance
        GameObject closestBall = null;
        float closestDistance = Mathf.Infinity;

        // Iterate through each ball in the array
        foreach (GameObject ball in allBalls)
        {
            // Calculate the distance between the ball and the human object
            float distanceToHuman = Vector3.Distance(ball.transform.position, humanPlayer.transform.position);

            // Check if this ball is closer to the human object than the previously closest ball
            if (distanceToHuman < closestDistance)
            {
                closestBall = ball;
                closestDistance = distanceToHuman;
            }
        }
        // Return the position of the closest ball
        if (closestBall != null && closestDistance <=4)
        {
            return closestBall.transform.position;
        }
        else
        {
            // No closest ball found, return Vector3.zero or any other default position
            return astroStation;
        }
    }

    public Vector3 convertStepIntoMovement(int step)
    {
      if(step == 0){astroOrientation = new Vector2(0,1); return new Vector3(pos_x,pos_y+1,0);} //Move Up
      if(step == 1){astroOrientation = new Vector2(0,-1); return new Vector3(pos_x,pos_y-1,0);} //Move Down
      if(step == 2){astroOrientation = new Vector2(-1,0); return new Vector3(pos_x-1,pos_y,0);} //Move Left
      if(step == 3){astroOrientation = new Vector2(1,0); return new Vector3(pos_x+1,pos_y,0);} //Move Right
      return new Vector3(0,0,0);
    }

    public void setNextOrientation(Vector3 step)
    { 
      if(step == new Vector3(pos_x, pos_y+1,0)){astroOrientation = new Vector2(0,1);} //Turn up
      if(step == new Vector3(pos_x, pos_y-1,0)){astroOrientation = new Vector2(0,-1);} //Turn Down
      if(step == new Vector3(pos_x-1,pos_y,0)){astroOrientation = new Vector2(-1,0);} //Turn Left
      if(step == new Vector3(pos_x+1,pos_y,0)){astroOrientation = new Vector2(1,0);} //Turn Right
    }

    //Check if human is currently holding a ball
    public bool humanHoldingBall(GameObject[] balls)
    {
      foreach (GameObject ball in balls)
      {   
        if(ball != null && !ball.GetComponent<SpriteRenderer>().enabled)
        {
          return true;
        }
      }
      return false;
    }

    public void moveOrRotateRobot(Vector3 newPosition, Vector2 newOrientation)
    { 
      if(animator.GetFloat("moveX")!=newOrientation.x || animator.GetFloat("moveY")!=newOrientation.y)
      {
        animator.SetFloat("moveX", newOrientation.x);
        animator.SetFloat("moveY", newOrientation.y);
      }
      if(transform.position != newPosition && ballInteraction.checkPositionVacancy(newPosition) && humanPlayer.transform.position != newPosition)
      {
        transform.position = Vector3.MoveTowards(transform.position, newPosition, movementSpeed * Time.deltaTime);
      }    
      
    }

    public bool arrayContains (GameObject[] array, GameObject objToCheck) 
    {
      foreach (GameObject obj in array) {if(obj == objToCheck) return true;}
      return false;
    }

    public (GameObject,float) FindClosestBall(GameObject[] balls, GameObject player)
    {

      // Initialize variables to keep track of the closest ball and its distance
      GameObject closestBall = null;
      float closestDistance = Mathf.Infinity;

      // Find the closest ball
      foreach (GameObject ball in balls)
      {   
        if (ball != null && ball.tag == "Ball")
        {
          float distance = Vector3.Distance(player.transform.position, ball.transform.position);
          if (distance < closestDistance)
          {
              closestDistance = distance;
              closestBall = ball;
          }
        }
      }
      //Debug.Log("Ball: " + closestBall.name + " pos: " + closestBall.transform.position);
      return (closestBall,closestDistance);
    }
}
