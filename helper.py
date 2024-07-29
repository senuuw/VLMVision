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


def get_video_length(file_path):
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print("Error opening video file")
            return None

        # Get frame rate and total frame count
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if fps > 0 and frame_count > 0:
            duration = frame_count / fps
            return duration
        else:
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        cap.release()

