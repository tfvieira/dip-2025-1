"""
load_image_from_url_solution.py

This script loads an image from a given URL and displays it using OpenCV.

Usage:
    python load_image_from_url_solution.py <image_url>

Examples:
    python load_image_from_url_solution.py https://raw.githubusercontent.com/tfvieira/dip-2025-1/main/img/aerial.tif
    python load_image_from_url_solution.py https://raw.githubusercontent.com/tfvieira/dip-2025-1/main/img/aerial.tif --grayscale
"""

import urllib.request
import cv2 as cv
import numpy as np
import argparse
import sys


def load_image_from_url(url: str, flags: int = cv.IMREAD_COLOR) -> np.ndarray:
    """
    Loads an image from a given URL using OpenCV's imdecode method.

    Parameters:
    - url (str): Direct link to the image file.
    - flags (int): OpenCV imread flags (default: cv.IMREAD_COLOR).

    Returns:
    - np.ndarray: The decoded image as a NumPy array.

    Raises:
    - RuntimeError: If the image cannot be loaded or decoded.
    """
    try:
        ### START CODE HERE ###
        ### TODO
        image = None
        ### END CODE HERE ###

        return image

    except Exception as e:
        raise RuntimeError(f"[ERROR] Could not load image from URL '{url}': {e}") from e


def main():
    parser = argparse.ArgumentParser(description="Load and display an image from a URL.")
    parser.add_argument("url", type=str, help="Direct URL to the image file.")
    parser.add_argument("--grayscale", action="store_true", help="Load the image in grayscale mode.")
    args = parser.parse_args()

    print(f"[INFO] Downloading image from: {args.url}")

    # Choose flag based on optional argument
    flag = cv.IMREAD_GRAYSCALE if args.grayscale else cv.IMREAD_COLOR

    try:
        img = load_image_from_url(args.url, flags=flag)
        print("[INFO] Image loaded successfully.")

        # Display the image
        cv.namedWindow('Image from URL', cv.WINDOW_KEEPRATIO)
        cv.imshow('Image from URL', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    except RuntimeError as err:
        print(err)
        sys.exit(1)


if __name__ == "__main__":
    main()
