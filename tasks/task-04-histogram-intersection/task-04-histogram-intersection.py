import numpy as np
import matplotlib.pyplot as plt
def compute_histogram_intersection(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute the histogram intersection similarity score between two grayscale images.

    This function calculates the similarity between the grayscale intensity 
    distributions of two images by computing the intersection of their 
    normalized 256-bin histograms.

    The histogram intersection is defined as the sum of the minimum values 
    in each corresponding bin of the two normalized histograms. The result 
    ranges from 0.0 (no overlap) to 1.0 (identical histograms).

    Parameters:
        img1 (np.ndarray): First input image as a 2D NumPy array (grayscale).
        img2 (np.ndarray): Second input image as a 2D NumPy array (grayscale).

    Returns:
        float: Histogram intersection score in the range [0.0, 1.0].

    Raises:
        ValueError: If either input is not a 2D array (i.e., not grayscale).
    """    
    if img1.ndim != 2 or img2.ndim != 2:
        raise ValueError("Both input images must be 2D grayscale arrays.")

    ### START CODE HERE ###
    # Step 1: initialize base image with 0.5
    intersection = 0.0
    
    hist1, _ = np.histogram(img1, bins=256, range=(0, 256), density=True)
    hist2, _ = np.histogram(img2, bins=256, range=(0, 256), density=True)
    
    intersection = np.sum(np.minimum(hist1, hist2))
    #plot histogram
    #plt.figure(figsize=(10, 5))
    #plt.subplot(1, 2, 1)
    #plt.title('Histogram of Image 1')
    #plt.bar(range(256), hist1, width=1, color='blue', alpha=0.5)
    #plt.subplot(1, 2, 2)
    #plt.title('Histogram of Image 2')
    #plt.bar(range(256), hist2, width=1, color='red', alpha=0.5)
    #plt.tight_layout()
    #plt.show()
    ### END CODE HERE ###


    return float(intersection)

