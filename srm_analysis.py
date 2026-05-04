import numpy as np
import cv2
from PIL import Image
import os

def get_srm_filters():
    """
    Defines basic SRM (Steganalysis Rich Model) filters to extract noise residuals.
    These filters are effective at highlighting high-frequency inconsistencies.
    """
    # 1st order filters
    f1 = np.array([[-1, 2, -1]])
    f2 = np.array([[-1], [2], [-1]])
    
    # 2nd order filters
    f3 = np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]])
    
    # Square filter
    f4 = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]])
    
    return [f1, f2, f3, f4]

def apply_srm(img_path):
    """
    Applies SRM filters to an image and returns a combined noise residual map.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    
    filters = get_srm_filters()
    residuals = []
    
    for f in filters:
        # Apply filter to each channel
        res = cv2.filter2D(img, -1, f)
        residuals.append(np.abs(res))
    
    # Combine residuals (average) and normalize for visualization
    combined_res = np.mean(residuals, axis=0)
    combined_res = np.clip(combined_res, 0, 255).astype(np.uint8)
    
    # Convert to PIL for easy handling
    return Image.fromarray(combined_res)

def get_srm_score(img_path):
    """
    A simple heuristic to detect tampering based on noise residual variance.
    Higher variance in specific regions often indicates manipulation.
    """
    img = cv2.imread(img_path, 0) # Grayscale
    if img is None: return 0
    
    # Apply a high-pass filter to get noise
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    variance = np.var(laplacian)
    
    # Normal images typically have a consistent noise variance
    # Tampered images often have higher or inconsistent variance
    return variance

if __name__ == "__main__":
    # Example usage
    test_img = "test_user_img.jpg"
    if os.path.exists(test_img):
        srm_img = apply_srm(test_img)
        srm_img.save("user_srm_result.jpg")
        print("SRM analysis complete. Noise residual saved to user_srm_result.jpg")
        print(f"Noise Variance Score: {get_srm_score(test_img):.2f}")
