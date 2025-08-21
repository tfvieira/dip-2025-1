# Load and Display Image from URL (OpenCV)

This exercise demonstrates how to **load an image directly from a URL** and display it using OpenCV.  
You will complete the function `load_image_from_url` to fetch the image over the internet, decode it, and return it as a NumPy array.

---

## 📌 Objective
Implement the missing part of:

```python
def load_image_from_url(url: str, flags: int = cv.IMREAD_COLOR) -> np.ndarray:
    """
    Loads an image from a given URL using OpenCV's imdecode method.
    """
