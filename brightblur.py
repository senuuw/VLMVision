import cv2
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
