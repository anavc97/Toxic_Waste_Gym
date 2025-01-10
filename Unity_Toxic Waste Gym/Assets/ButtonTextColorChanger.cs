using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;
using UnityEngine.Video;
using TMPro; // If using TextMeshPro

public class ButtonTextColorChanger : MonoBehaviour, IPointerEnterHandler, IPointerExitHandler
{
    public Button button;
    public TextMeshProUGUI buttonText;
    public Color normalTextColor = new Color32(243, 210, 0, 255); // Hex: F3D200
    public Color highlightedTextColor = new Color32(243, 170, 0, 255); // Hex: F37A00

    private void Start()
    {           

        if (button == null)
        {
            button = GetComponent<Button>();
        }

        if (buttonText == null)
        {
            buttonText = button.GetComponentInChildren<TextMeshProUGUI>();
        }

        // Set initial text color
        buttonText.color = normalTextColor;
    }

    public void OnPointerEnter(PointerEventData eventData)
    {
        buttonText.color = highlightedTextColor;
    }

    public void OnPointerExit(PointerEventData eventData)
    {
        buttonText.color = normalTextColor;
    }
}
