import cv2
import numpy as np

def shadow_removal(image):
    """
    Melakukan shadow removal konvensional pada citra dokumen.
    """

    # Resize agar seragam
    image = cv2.resize(image, (600, 800))

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Estimate background menggunakan morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    background = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)

    background = np.where(background == 0, 1, background)

    # Illumination correction
    corrected = cv2.divide(gray, background, scale=255)
    corrected_uint8 = (corrected * 255).astype(np.uint8)

    # Adaptive thresholding
    result = cv2.adaptiveThreshold(
        corrected_uint8,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        10
    )

    return image, corrected_uint8, result
