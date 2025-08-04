# Histogram Intersection – Image Similarity Task

## Objective

Your task is to compute the **Histogram Intersection** between two grayscale images.

This metric measures how much the grayscale intensity distributions (histograms) of the two images overlap. A value of `1.0` means the histograms are identical. A value of `0.0` means there is no overlap at all.

---

## Steps

Implement:

$$
\text{Intersection}(H_1, H_2) = \sum_{i=1}^{256} \min(H_1[i], H_2[i])
$$

---

## Function Signature

Edit the `student_solution.py` file and complete the following function:

```python
def compute_histogram_intersection(image_path1: str, image_path2: str) -> float:
    pass
