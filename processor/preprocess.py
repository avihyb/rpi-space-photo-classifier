import cv2
import numpy as np


def is_dark(image: np.ndarray, threshold: float = 40.0) -> bool:
    """Return True if the image is mostly dark."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) < threshold


def is_blurry(image: np.ndarray, threshold: float = 100.0) -> bool:
    """Return True if the image appears blurry using the Laplacian variance metric."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


def auto_crop(image: np.ndarray, min_area_ratio: float = 0.1) -> np.ndarray:
    """Crop to the brightest region if the image is mostly dark."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # apply Otsu threshold to find bright areas
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    # choose largest contour
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if (w * h) < min_area_ratio * (gray.shape[0] * gray.shape[1]):
        return image
    return image[y : y + h, x : x + w]
