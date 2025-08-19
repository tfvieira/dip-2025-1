import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_image_info(image):
    """
    Extracts metadata and statistical information from an image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image (grayscale or color).

    Returns
    -------
    dict
        Dictionary containing image metadata and statistics.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a NumPy array representing an image.")

    # Extract image properties
    height, width = image.shape[:2]
    depth = 1 if len(image.shape) == 2 else image.shape[2]  # Number of channels
    dtype = image.dtype
    nbytes = image.nbytes

    # Create result dictionary
    info = {
        "width": width,
        "height": height,
        "dtype": str(dtype),
        "depth": depth,
        "nbytes": nbytes,
        "statistics": {}
    }

    ### START CODE HERE ###
    # Handle grayscale vs multi-channel separately
    if depth == 1:
        # Single-channel
        info["statistics"]["min"] = None
        info["statistics"]["max"] = None
        info["statistics"]["mean"] = None
        info["statistics"]["std_dev"] = None
        info["statistics"]["median"] = None
        hist, bins = np.histogram(image, bins=256, range=(0, 256))
        info["statistics"]["histogram"] = hist.tolist()
    else:
        # Multi-channel: compute per channel
        channels = cv2.split(image)
        for idx, ch in enumerate(channels):
            ch_stats = {
                "min": None,
                "max": None,
                "mean": None,
                "std_dev": None,
                "median": None,
            }
            hist, bins = np.histogram(ch, bins=256, range=(0, 256))
            ch_stats["histogram"] = hist.tolist()
            info["statistics"][f"channel_{idx}"] = ch_stats
    ### END CODE HERE ###

    return info


# ===============================
# ====== TEST EXAMPLES ==========
# ===============================
if __name__ == "__main__":

    # Example 1: Grayscale synthetic gradient image
    gray_img = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    gray_info = get_image_info(gray_img)

    # Example 2: Binary pattern (checkerboard)
    checker = np.indices((8, 8)).sum(axis=0) % 2 * 255
    checker = checker.astype(np.uint8)
    checker_info = get_image_info(checker)

    # Example 3: Random RGB image
    # RGB image from Gaussian distribution (mean=128, std=20)
    rgb_img = 128 + 20 * np.random.randn(100, 100, 3)
    # Clip to valid pixel range [0, 255]
    rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)
    rgb_info = get_image_info(rgb_img)

    print("Grayscale Image Info:", gray_info)
    print("\nCheckerboard Image Info:", checker_info)
    print("\nRandom RGB Image Info:", rgb_info)

    # ====== Visualization with histograms ======
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    # Top row = images
    axes[0, 0].imshow(gray_img, cmap="gray")
    axes[0, 0].set_title("Grayscale Gradient")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(checker, cmap="gray")
    axes[0, 1].set_title("Checkerboard")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("Random RGB")
    axes[0, 2].axis("off")

    # Bottom row = histograms
    axes[1, 0].plot(gray_info["statistics"]["histogram"], color="black")
    axes[1, 0].set_title("Histogram (Grayscale)")

    axes[1, 1].plot(checker_info["statistics"]["histogram"], color="black")
    axes[1, 1].set_title("Histogram (Checkerboard)")

    # For RGB, plot each channel
    for idx, color in zip(range(3), ["b", "g", "r"]):
        axes[1, 2].plot(rgb_info["statistics"][f"channel_{idx}"]["histogram"], color=color)
    axes[1, 2].set_title("Histogram (RGB)")

    plt.tight_layout()
    plt.show()
