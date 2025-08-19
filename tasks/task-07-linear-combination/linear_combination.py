import cv2 as cv
import numpy as np


def linear_combination(i1: np.ndarray, i2: np.ndarray, a1: float, a2: float) -> np.ndarray:
    """
    Compute the linear combination of two images using OpenCV: 
    i_out = a1 * i1 + a2 * i2.

    Args:
        i1 (np.ndarray): First input image.
        i2 (np.ndarray): Second input image.
        a1 (float): Scalar weight for the first image.
        a2 (float): Scalar weight for the second image.

    Returns:
        np.ndarray: The resulting blended image with the same dtype as the input images.

    Raises:
        ValueError: If the input images do not have the same dimensions.
    """
    if i1.shape != i2.shape:
        raise ValueError("Input images must have the same dimensions.")

    ### START CODE HERE ###
    ### TODO
    i_out = None
    ### END CODE HERE ###

    return i_out


if __name__ == "__main__":
    """
    Example usage of the linear_combination function with interactive trackbar control.

    This script:
        1. Loads two images from disk ('dog.png' and 'bird.png').
        2. Creates OpenCV windows for each image and a result window.
        3. Initializes a trackbar that controls the blending factor alpha in the range [0, 1].
        4. Continuously updates the linear combination of the two images based on the trackbar value.
        5. Displays the original images and the blended output in real time.
        6. Allows the user to quit by pressing the 'q' key.

    Notes:
        - Both images must exist in the working directory and have the same size and channels.
        - The blending is defined as: output = alpha * i1 + (1 - alpha) * i2.
        - Windows remain responsive until the user exits.
    """

    # Load images
    i1 = cv.imread('dog.png', cv.IMREAD_COLOR)
    i2 = cv.imread('bird.png', cv.IMREAD_COLOR)

    if i1 is None or i2 is None:
        raise FileNotFoundError("One or both images could not be loaded. Check file paths.")

    # Create windows
    cv.namedWindow('I1', cv.WINDOW_KEEPRATIO)
    cv.namedWindow('I2', cv.WINDOW_KEEPRATIO)
    window_name = "Linear Combination"
    cv.namedWindow(window_name, cv.WINDOW_KEEPRATIO)

    # Initial alpha value
    alpha_init = 50  # range: 0–100. Divide by 100 later.
    cv.createTrackbar("Alpha", window_name, alpha_init, 100, lambda x: None)

    while True:
        # Get current trackbar value
        alpha = cv.getTrackbarPos("Alpha", window_name) / 100.0
        beta = 1.0 - alpha

        # Blend images
        output = linear_combination(i1, i2, alpha, beta)

        # Show result
        cv.imshow('I1', i1)
        cv.imshow('I2', i2)
        cv.imshow(window_name, output)

        # Break with 'q'
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv.destroyAllWindows()
