import cv2
import numpy as np

def detect_blur_and_bright_spot(image_path):
    # Read the image
    image = cv2.imread(image_path)
    # Apply Laplacian filter for edge detection
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    laplacian_variance = round(laplacian.var(), 2)
    binary_variance = round(binary_image.var(), 2)
    return laplacian_variance, binary_variance

def calculate_exposure_percentages(image_path, threshold=10):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    total_pixels = image.size
    # Calculate the number of underexposed pixels
    underexposed_pixels = np.sum(image < threshold)
    # Calculate the number of overexposed pixels
    overexposed_pixels = np.sum(image > 255 - threshold)
    # Calculate the percentages
    under_percentage = (underexposed_pixels / total_pixels) * 100
    over_percentage = (overexposed_pixels / total_pixels) * 100

    return under_percentage, over_percentage

