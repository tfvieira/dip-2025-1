# Image Similarity Exercise

This exercise is about implementing basic **image similarity metrics** using only **NumPy**.  
You will compute four metrics commonly used to compare grayscale images:

1. **Mean Squared Error (MSE)**
2. **Peak Signal-to-Noise Ratio (PSNR)**
3. **Structural Similarity Index (SSIM)** (simplified version)
4. **Normalized Pearson Correlation Coefficient (NPCC)**

Both input images (`i1`, `i2`) are assumed to be **grayscale** and **normalized** (values between `0` and `1`).

---

## 1. Mean Squared Error (MSE)

The **MSE** measures the average squared difference between the pixels of the two images.

$$
\text{MSE}(i_1, i_2) = \frac{1}{N} \sum_{x=1}^{H} \sum_{y=1}^{W} \left( i_1(x,y) - i_2(x,y) \right)^2
$$

Where:
- $ H, W $ are the image height and width.
- $ N = H \times W $ is the total number of pixels.

A **lower MSE** means the images are more similar.  
A value of **0** means the images are identical.

---

## 2. Peak Signal-to-Noise Ratio (PSNR)

The **PSNR** is derived from the MSE and expresses the similarity in terms of a **logarithmic decibel (dB) scale**.  
It is commonly used in image compression quality assessment.

$$
\text{PSNR}(i_1, i_2) = 10 \cdot \log_{10}\left(\frac{L^2}{\text{MSE}(i_1, i_2)}\right)
$$

Where:
- $ L $ is the maximum possible pixel value (here $ L = 1 $, since the images are normalized).
- $\text{MSE}(i_1, i_2)$ is as defined above.

- Higher PSNR → better similarity.
- If MSE = 0, PSNR is **infinite** (perfect match).

---

## 3. Structural Similarity Index (SSIM) — Simplified

The **SSIM** measures structural similarity by comparing **luminance**, **contrast**, and **structure**.  
A simplified global version can be computed using image means and variances:

$$
\text{SSIM}(i_1, i_2) = \frac{(2 \mu_1 \mu_2 + C_1)(2 \sigma_{12} + C_2)}
{(\mu_1^2 + \mu_2^2 + C_1)(\sigma_1^2 + \sigma_2^2 + C_2)}
$$

Where:
- $ \mu_1, \mu_2 $ = mean pixel values of $ i_1 $ and $ i_2 $.
- $ \sigma_1^2, \sigma_2^2 $ = variances of $ i_1 $ and $ i_2 $.
- $ \sigma_{12} $ = covariance between $ i_1 $ and $ i_2 $.
- $ C_1, C_2 $ = small constants to stabilize the division (e.g., $10^{-8}$).

- SSIM ranges between **-1 and 1** (typically close to 0–1).
- **1 means perfect structural similarity.**

---

## 4. Normalized Pearson Correlation Coefficient (NPCC)

The **NPCC** measures how strongly two images are linearly correlated.

$$
\text{NPCC}(i_1, i_2) = \frac{\sum (i_1 - \mu_1)(i_2 - \mu_2)}
{\sqrt{\sum (i_1 - \mu_1)^2} \cdot \sqrt{\sum (i_2 - \mu_2)^2}}
$$

Where:
- $ \mu_1, \mu_2 $ are the means of $ i_1 $ and $ i_2 $.

- NPCC ranges between **-1 and 1**:
  - **1** → perfect positive correlation.
  - **0** → no correlation.
  - **-1** → perfect negative correlation.

---
