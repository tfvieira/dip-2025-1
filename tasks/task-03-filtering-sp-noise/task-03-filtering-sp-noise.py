import cv2 as cv
import numpy as np
def median_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Applies a median filter to the input image.

    Parameters:
        image (np.ndarray): Input image (grayscale).
        kernel_size (int): Size of the median filter kernel.

    Returns:
        np.ndarray: Filtered image.
    """
    pad_size = kernel_size // 2
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode='edge')
    filtered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the kernel region
            kernel_region = padded_image[i:i + kernel_size, j:j + kernel_size]
            # Compute the median of the kernel region
            filtered_image[i, j] = np.median(kernel_region)
    return filtered_image 

def remove_salt_and_pepper_noise(image: np.ndarray) -> np.ndarray:
    """
    Removes salt and pepper noise from a grayscale image.

    Parameters:
        image (np.ndarray): Noisy input image (grayscale).

    Returns:
        np.ndarray: Denoised image.
    """
    #image_filtered = cv.medianBlur(image, 5)  # Example of a simple median filter
    # TODO: Implement noise removal here (e.g., median filtering)
    image_filtered = median_filter(image, kernel_size=7)
    # Display the original and filtered images
    #cv.imshow('Filtered Image',image)
    #cv.waitKey(0)
    #cv.imshow('Filtered Image',image_filtered)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    return image  # Replace this with your filtering implementation

if __name__ == "__main__":
    noisy_image = cv.imread("noisy_image.png", cv.IMREAD_GRAYSCALE)
    denoised_image = remove_salt_and_pepper_noise(noisy_image)
    cv.imwrite("denoised_image.png", denoised_image)
