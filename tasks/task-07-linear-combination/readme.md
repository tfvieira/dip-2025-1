# Image Blending with Linear Combination (OpenCV)

This exercise demonstrates how to blend two images using a **linear combination** of pixel values.  
You will implement the missing part of the function `linear_combination` and use an interactive trackbar to adjust the blending ratio (`alpha`) in real time.

---

## 📌 Objective
Implement the function:

```python
def linear_combination(i1, i2, a1, a2):
    """
    Compute the linear combination of two images:
    i_out = a1 * i1 + a2 * i2
    """
