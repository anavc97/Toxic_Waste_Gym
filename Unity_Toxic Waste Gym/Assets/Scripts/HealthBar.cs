using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HealthBar : MonoBehaviour
{   
    public Transform bar;

    public void Start()
    {
        bar = transform.Find("Bar");
    }

    public void SetSize(float sizeNormalized)
    {   
        Debug.Log(bar.localScale);

        bar.localScale = new Vector3(sizeNormalized,1f);
    }
}
